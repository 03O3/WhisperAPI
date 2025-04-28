package main

import (
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"time"

	"github.com/gin-gonic/gin"
)

var (
	// Конфигурация
	whisperHost = getEnv("WHISPER_HOST", "127.0.0.1")
	whisperPort = getEnv("WHISPER_PORT", "9000")
	serverPort  = getEnv("SERVER_PORT", "8080")
)

// Глобальный клиент Whisper
var whisperClient *WhisperClient

// Вспомогательная функция для получения переменных окружения с значениями по умолчанию
func getEnv(key, defaultValue string) string {
	value := os.Getenv(key)
	if value == "" {
		return defaultValue
	}
	return value
}

// Инициализация сервера
func init() {
	// Инициализируем клиент для связи с сервисом Whisper
	whisperClient = NewWhisperClient(whisperHost, whisperPort)
}

// Основная функция веб-сервера
func main() {
	router := gin.Default()

	// Устанавливаем ограничение на размер загружаемых файлов (20MB)
	router.MaxMultipartMemory = 20 << 20

	// Middleware для обработки CORS
	router.Use(func(c *gin.Context) {
		c.Writer.Header().Set("Access-Control-Allow-Origin", "*")
		c.Writer.Header().Set("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
		c.Writer.Header().Set("Access-Control-Allow-Headers", "Content-Type, Content-Length, Accept-Encoding, Authorization")

		if c.Request.Method == "OPTIONS" {
			c.AbortWithStatus(http.StatusNoContent)
			return
		}

		c.Next()
	})

	// Группа эндпоинтов API
	api := router.Group("/api")
	{
		// Получение списка моделей
		api.GET("/models", getModelsHandler)

		// Транскрипция аудиофайла
		api.POST("/transcribe", transcribeHandler)

		// Информация о статусе сервера
		api.GET("/health", healthCheckHandler)
	}

	// Статические файлы для веб-интерфейса
	router.Static("/static", "./static")
	router.StaticFile("/", "./static/index.html")

	// Запуск сервера
	log.Printf("Запуск сервера на порту %s...", serverPort)
	if err := router.Run(":" + serverPort); err != nil {
		log.Fatalf("Ошибка при запуске сервера: %v", err)
	}
}

// Обработчик для проверки здоровья сервиса
func healthCheckHandler(c *gin.Context) {
	c.JSON(http.StatusOK, gin.H{
		"status":      "ok",
		"server_time": time.Now(),
		"version":     "1.0.0",
	})
}

// Обработчик для получения списка моделей
func getModelsHandler(c *gin.Context) {
	models, err := whisperClient.ListModels()
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{
			"error": fmt.Sprintf("Ошибка при получении списка моделей: %v", err),
		})
		return
	}

	c.JSON(http.StatusOK, models)
}

// Обработчик для транскрипции аудио
func transcribeHandler(c *gin.Context) {
	// Получаем загруженный файл
	file, err := c.FormFile("file")
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{
			"error": "Файл не найден в запросе",
		})
		return
	}

	// Получаем параметры из запроса
	model := c.DefaultPostForm("model", "base")
	var language *string
	if langValue := c.PostForm("language"); langValue != "" {
		language = &langValue
	}
	task := c.DefaultPostForm("task", "transcribe")

	// Открываем загруженный файл
	src, err := file.Open()
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{
			"error": fmt.Sprintf("Ошибка при открытии файла: %v", err),
		})
		return
	}
	defer src.Close()

	// Читаем содержимое файла в память
	audioData, err := io.ReadAll(src)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{
			"error": fmt.Sprintf("Ошибка при чтении файла: %v", err),
		})
		return
	}

	// Выполняем транскрипцию, передавая данные напрямую
	startTime := time.Now()
	result, err := whisperClient.TranscribeData(audioData, model, language, task)
	elapsedTime := time.Since(startTime)

	// Проверяем ошибки
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{
			"error": fmt.Sprintf("Ошибка при выполнении транскрипции: %v", err),
		})
		return
	}

	// Возвращаем результат с дополнительной информацией о времени обработки
	result.ProcessingTime = elapsedTime.Seconds()
	c.JSON(http.StatusOK, result)
}
