package main

import (
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"errors"
	"io"
	"log"
	"net"
	"path/filepath"
	"time"
)

// WhisperClient - клиент для взаимодействия с Python сервисом
type WhisperClient struct {
	Host     string
	Port     string
	conn     net.Conn
	connLock chan struct{}
}

// Константы для работы с соединением
const (
	headerSize   = 8
	maxRetries   = 3
	retryTimeout = 2 * time.Second
)

// Структуры для запросов и ответов
type TranscriptionRequest struct {
	Command   string  `json:"command"`
	AudioPath string  `json:"audio_path,omitempty"`
	AudioData string  `json:"audio_data,omitempty"`
	Model     string  `json:"model,omitempty"`
	Language  *string `json:"language,omitempty"`
	Task      string  `json:"task,omitempty"`
}

type TranscriptionResponse struct {
	Text           string                   `json:"text,omitempty"`
	Language       string                   `json:"language,omitempty"`
	Segments       []map[string]interface{} `json:"segments,omitempty"`
	ProcessingTime float64                  `json:"processing_time,omitempty"`
	Error          string                   `json:"error,omitempty"`
}

type ModelsRequest struct {
	Command string `json:"command"`
}

type ModelsResponse struct {
	AvailableModels map[string]string `json:"available_models,omitempty"`
	LoadedModels    []string          `json:"loaded_models,omitempty"`
	Error           string            `json:"error,omitempty"`
}

// NewWhisperClient создает нового клиента для работы с Whisper сервисом
func NewWhisperClient(host, port string) *WhisperClient {
	client := &WhisperClient{
		Host:     host,
		Port:     port,
		connLock: make(chan struct{}, 1),
	}
	// Инициализация мьютекса
	client.connLock <- struct{}{}
	return client
}

// ensureConnection устанавливает соединение с сервером, если оно отсутствует
func (c *WhisperClient) ensureConnection() error {
	if c.conn != nil {
		return nil
	}

	var err error
	for i := 0; i < maxRetries; i++ {
		addr := net.JoinHostPort(c.Host, c.Port)
		c.conn, err = net.Dial("tcp", addr)
		if err == nil {
			return nil
		}

		log.Printf("Не удалось установить соединение (попытка %d/%d): %v", i+1, maxRetries, err)
		if i < maxRetries-1 {
			time.Sleep(retryTimeout)
		}
	}

	return errors.New("не удалось установить соединение с Whisper сервисом")
}

// closeConnection закрывает текущее соединение
func (c *WhisperClient) closeConnection() {
	if c.conn != nil {
		c.conn.Close()
		c.conn = nil
	}
}

// sendRequest отправляет запрос и получает ответ от сервера
func (c *WhisperClient) sendRequest(requestData interface{}) ([]byte, error) {
	// Блокируем доступ к соединению
	<-c.connLock
	defer func() { c.connLock <- struct{}{} }()

	// Убеждаемся, что соединение установлено
	if err := c.ensureConnection(); err != nil {
		return nil, err
	}

	// Сериализуем запрос в JSON
	requestJSON, err := json.Marshal(requestData)
	if err != nil {
		return nil, err
	}

	// Подготавливаем заголовок с длиной сообщения
	requestLen := len(requestJSON)
	header := make([]byte, headerSize)
	binary.BigEndian.PutUint64(header, uint64(requestLen))

	// Отправляем заголовок и тело запроса
	if _, err := c.conn.Write(header); err != nil {
		c.closeConnection()
		return nil, err
	}
	if _, err := c.conn.Write(requestJSON); err != nil {
		c.closeConnection()
		return nil, err
	}

	// Получаем ответ
	headerBuf := make([]byte, headerSize)
	if _, err := io.ReadFull(c.conn, headerBuf); err != nil {
		c.closeConnection()
		return nil, err
	}

	responseLen := binary.BigEndian.Uint64(headerBuf)
	responseBuf := make([]byte, responseLen)
	if _, err := io.ReadFull(c.conn, responseBuf); err != nil {
		c.closeConnection()
		return nil, err
	}

	return responseBuf, nil
}

// Transcribe выполняет транскрипцию аудиофайла
func (c *WhisperClient) Transcribe(audioPath string, model string, language *string, task string) (*TranscriptionResponse, error) {
	// Преобразуем путь к абсолютному, чтобы Python сервис мог найти файл
	absPath, err := filepath.Abs(audioPath)
	if err != nil {
		return nil, errors.New("не удалось получить абсолютный путь к файлу: " + err.Error())
	}

	request := TranscriptionRequest{
		Command:   "transcribe",
		AudioPath: absPath,
		Model:     model,
		Language:  language,
		Task:      task,
	}

	responseData, err := c.sendRequest(request)
	if err != nil {
		return nil, err
	}

	var response TranscriptionResponse
	if err := json.Unmarshal(responseData, &response); err != nil {
		return nil, err
	}

	if response.Error != "" {
		return nil, errors.New(response.Error)
	}

	return &response, nil
}

// TranscribeData выполняет транскрипцию аудиоданных
func (c *WhisperClient) TranscribeData(audioData []byte, model string, language *string, task string) (*TranscriptionResponse, error) {
	// Используем base64 для безопасной передачи бинарных данных
	encodedData := base64.StdEncoding.EncodeToString(audioData)

	request := TranscriptionRequest{
		Command:   "transcribe",
		AudioData: encodedData,
		Model:     model,
		Language:  language,
		Task:      task,
	}

	responseData, err := c.sendRequest(request)
	if err != nil {
		return nil, err
	}

	var response TranscriptionResponse
	if err := json.Unmarshal(responseData, &response); err != nil {
		return nil, err
	}

	if response.Error != "" {
		return nil, errors.New(response.Error)
	}

	return &response, nil
}

// ListModels получает список доступных моделей
func (c *WhisperClient) ListModels() (*ModelsResponse, error) {
	request := ModelsRequest{
		Command: "list_models",
	}

	responseData, err := c.sendRequest(request)
	if err != nil {
		return nil, err
	}

	var response ModelsResponse
	if err := json.Unmarshal(responseData, &response); err != nil {
		return nil, err
	}

	if response.Error != "" {
		return nil, errors.New(response.Error)
	}

	return &response, nil
}

// Close закрывает соединение с сервером
func (c *WhisperClient) Close() {
	<-c.connLock
	defer func() { c.connLock <- struct{}{} }()
	c.closeConnection()
}
