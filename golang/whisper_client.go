package main

import (
	"context"
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"sync"
	"sync/atomic"
	"time"
)

// WhisperClient - клиент для взаимодействия с Python сервисом
type WhisperClient struct {
	Host     string
	Port     string
	conn     net.Conn
	connLock sync.Mutex
	metrics  Metrics
	client   *http.Client
}

// Metrics - метрики клиента
type Metrics struct {
	RequestsTotal    int64
	ErrorsTotal      int64
	ProcessingTimeMs int64
}

// Константы для работы с соединением
const (
	headerSize     = 8
	maxRetries     = 3
	retryTimeout   = 2 * time.Second
	connectTimeout = 5 * time.Second
)

// Структуры для запросов и ответов
type FileTranscriptionRequest struct {
	Command   string  `json:"command"`
	AudioPath string  `json:"audio_path"`
	Model     string  `json:"model,omitempty"`
	Language  *string `json:"language,omitempty"`
	Task      string  `json:"task,omitempty"`
}

type DataTranscriptionRequest struct {
	Command   string  `json:"command"`
	AudioData string  `json:"audio_data"`
	Model     string  `json:"model,omitempty"`
	Language  *string `json:"language,omitempty"`
	Task      string  `json:"task,omitempty"`
}

type Segment struct {
	Text  string  `json:"text"`
	Start float64 `json:"start"`
	End   float64 `json:"end"`
}

type TranscriptionResponse struct {
	Text           string    `json:"text,omitempty"`
	Language       string    `json:"language,omitempty"`
	Segments       []Segment `json:"segments,omitempty"`
	ProcessingTime float64   `json:"processing_time,omitempty"`
	Error          string    `json:"error,omitempty"`
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
	client := &http.Client{
		Timeout: 30 * time.Minute,
		Transport: &http.Transport{
			DialContext: (&net.Dialer{
				Timeout:   30 * time.Second,
				KeepAlive: 30 * time.Second,
			}).DialContext,
			MaxIdleConns:          100,
			IdleConnTimeout:       90 * time.Second,
			TLSHandshakeTimeout:   10 * time.Second,
			ExpectContinueTimeout: 1 * time.Second,
			ResponseHeaderTimeout: 30 * time.Second,
		},
	}
	return &WhisperClient{
		Host:   host,
		Port:   port,
		client: client,
	}
}

// ensureConnection устанавливает соединение с сервером, если оно отсутствует
func (c *WhisperClient) ensureConnection() error {
	c.connLock.Lock()
	defer c.connLock.Unlock()

	if c.conn != nil {
		return nil
	}

	var err error
	for i := 0; i < maxRetries; i++ {
		addr := net.JoinHostPort(c.Host, c.Port)
		dialer := &net.Dialer{
			Timeout: connectTimeout,
		}
		c.conn, err = dialer.Dial("tcp", addr)
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
	c.connLock.Lock()
	defer c.connLock.Unlock()

	if c.conn != nil {
		c.conn.Close()
		c.conn = nil
	}
}

// handleResponse обрабатывает ответ от сервера
func (c *WhisperClient) handleResponse(responseData []byte) (*TranscriptionResponse, error) {
	var response TranscriptionResponse
	if err := json.Unmarshal(responseData, &response); err != nil {
		atomic.AddInt64(&c.metrics.ErrorsTotal, 1)
		return nil, err
	}

	if response.Error != "" {
		atomic.AddInt64(&c.metrics.ErrorsTotal, 1)
		return nil, errors.New(response.Error)
	}

	return &response, nil
}

// sendRequest отправляет запрос и получает ответ от сервера
func (c *WhisperClient) sendRequest(requestData interface{}) ([]byte, error) {
	startTime := time.Now()
	atomic.AddInt64(&c.metrics.RequestsTotal, 1)

	defer func() {
		duration := time.Since(startTime)
		atomic.AddInt64(&c.metrics.ProcessingTimeMs, duration.Milliseconds())
		c.logRequest("sendRequest", duration, nil)
	}()

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

	// Устанавливаем таймаут для записи
	c.conn.SetWriteDeadline(time.Now().Add(connectTimeout))
	if _, err := c.conn.Write(header); err != nil {
		c.closeConnection()
		return nil, err
	}
	if _, err := c.conn.Write(requestJSON); err != nil {
		c.closeConnection()
		return nil, err
	}
	c.conn.SetWriteDeadline(time.Time{})

	// Получаем ответ
	headerBuf := make([]byte, headerSize)
	c.conn.SetReadDeadline(time.Now().Add(connectTimeout))
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
	c.conn.SetReadDeadline(time.Time{})

	return responseBuf, nil
}

// logRequest логирует информацию о запросе
func (c *WhisperClient) logRequest(command string, duration time.Duration, err error) {
	if err != nil {
		log.Printf("Request failed: command=%s duration=%v error=%v", command, duration, err)
	} else {
		log.Printf("Request succeeded: command=%s duration=%v", command, duration)
	}
}

// Transcribe выполняет транскрипцию аудиофайла
func (c *WhisperClient) Transcribe(audioPath string, model string, language *string, task string) (*TranscriptionResponse, error) {
	// Преобразуем путь к абсолютному, чтобы Python сервис мог найти файл
	absPath, err := filepath.Abs(audioPath)
	if err != nil {
		return nil, fmt.Errorf("не удалось получить абсолютный путь к файлу: %v", err)
	}

	// Проверяем существование файла
	if _, err := os.Stat(absPath); os.IsNotExist(err) {
		return nil, fmt.Errorf("файл не существует: %s", absPath)
	}

	// Проверяем язык
	if language != nil && *language == "" {
		language = nil
	}

	request := FileTranscriptionRequest{
		Command:   "transcribe",
		AudioPath: absPath,
		Model:     model,
		Language:  language,
		Task:      task,
	}

	// Увеличиваем таймаут для больших файлов
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Minute)
	defer cancel()

	responseData, err := c.sendRequestWithContext(ctx, request)
	if err != nil {
		return nil, fmt.Errorf("ошибка при отправке запроса: %v", err)
	}

	return c.handleResponse(responseData)
}

// sendRequestWithContext отправляет запрос с поддержкой контекста
func (c *WhisperClient) sendRequestWithContext(_ context.Context, request interface{}) ([]byte, error) {
	startTime := time.Now()
	atomic.AddInt64(&c.metrics.RequestsTotal, 1)

	defer func() {
		duration := time.Since(startTime)
		atomic.AddInt64(&c.metrics.ProcessingTimeMs, duration.Milliseconds())
		c.logRequest("sendRequest", duration, nil)
	}()

	// Сериализуем запрос в JSON
	requestJSON, err := json.Marshal(request)
	if err != nil {
		return nil, fmt.Errorf("ошибка при сериализации запроса: %v", err)
	}

	// Устанавливаем соединение
	conn, err := net.DialTimeout("tcp", net.JoinHostPort(c.Host, c.Port), 30*time.Second)
	if err != nil {
		return nil, fmt.Errorf("ошибка при установке соединения: %v", err)
	}
	defer conn.Close()

	// Устанавливаем таймауты
	conn.SetDeadline(time.Now().Add(30 * time.Minute))

	// Отправляем длину сообщения
	header := make([]byte, 8)
	binary.BigEndian.PutUint64(header, uint64(len(requestJSON)))
	if _, err := conn.Write(header); err != nil {
		return nil, fmt.Errorf("ошибка при отправке заголовка: %v", err)
	}

	// Отправляем данные
	if _, err := conn.Write(requestJSON); err != nil {
		return nil, fmt.Errorf("ошибка при отправке данных: %v", err)
	}

	// Читаем ответ
	headerBuf := make([]byte, 8)
	if _, err := io.ReadFull(conn, headerBuf); err != nil {
		return nil, fmt.Errorf("ошибка при чтении заголовка ответа: %v", err)
	}

	responseLen := binary.BigEndian.Uint64(headerBuf)
	responseBuf := make([]byte, responseLen)
	if _, err := io.ReadFull(conn, responseBuf); err != nil {
		return nil, fmt.Errorf("ошибка при чтении ответа: %v", err)
	}

	return responseBuf, nil
}

// TranscribeWithContext выполняет транскрипцию с поддержкой контекста
func (c *WhisperClient) TranscribeWithContext(ctx context.Context, audioPath string, model string, language *string, task string) (*TranscriptionResponse, error) {
	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	default:
		return c.Transcribe(audioPath, model, language, task)
	}
}

// TranscribeData выполняет транскрипцию аудиоданных
func (c *WhisperClient) TranscribeData(audioData []byte, model string, language *string, task string) (*TranscriptionResponse, error) {
	// Проверяем язык
	if language != nil && *language == "" {
		language = nil
	}

	// Используем base64 для безопасной передачи бинарных данных
	encodedData := base64.StdEncoding.EncodeToString(audioData)

	request := DataTranscriptionRequest{
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

	return c.handleResponse(responseData)
}

// TranscribeDataWithContext выполняет транскрипцию данных с поддержкой контекста
func (c *WhisperClient) TranscribeDataWithContext(ctx context.Context, audioData []byte, model string, language *string, task string) (*TranscriptionResponse, error) {
	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	default:
		return c.TranscribeData(audioData, model, language, task)
	}
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

// GetMetrics возвращает текущие метрики клиента
func (c *WhisperClient) GetMetrics() Metrics {
	return c.metrics
}

// Close закрывает соединение с сервером
func (c *WhisperClient) Close() {
	c.closeConnection()
}
