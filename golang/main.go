package main

import (
	"fmt"
	"log"
	"net/http"
	"os"
	"time"

	"github.com/gin-contrib/cors"
	"github.com/gin-gonic/gin"
	"github.com/rs/zerolog"
)

var (
	// Конфигурация
	whisperHost = getEnv("WHISPER_HOST", "127.0.0.1")
	whisperPort = getEnv("WHISPER_PORT", "9000")
	serverPort  = getEnv("SERVER_PORT", "8080")
	// Режим работы (debug/release)
	ginMode = getEnv("GIN_MODE", "debug")
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
	// Устанавливаем режим работы Gin
	gin.SetMode(ginMode)

	// Инициализируем клиент для связи с сервисом Whisper
	whisperClient = NewWhisperClient(whisperHost, whisperPort)
}

// Основная функция веб-сервера
func main() {
	// Настраиваем логирование
	logger := zerolog.New(os.Stdout).With().Timestamp().Logger()
	gin.DefaultWriter = logger

	// Создаем роутер с настройками
	router := gin.New()

	// Middleware
	router.Use(gin.Recovery())
	router.Use(cors.New(cors.Config{
		AllowOrigins:     []string{"*"},
		AllowMethods:     []string{"GET", "POST", "OPTIONS"},
		AllowHeaders:     []string{"Origin", "Content-Type", "Accept", "Authorization"},
		ExposeHeaders:    []string{"Content-Length"},
		AllowCredentials: true,
		MaxAge:           12 * time.Hour,
	}))
	router.Use(gin.LoggerWithFormatter(func(param gin.LogFormatterParams) string {
		return fmt.Sprintf("[%s] | %3d | %13v | %15s | %-7s %s\n%s",
			param.TimeStamp.Format("2006/01/02 - 15:04:05"),
			param.StatusCode,
			param.Latency,
			param.ClientIP,
			param.Method,
			param.Path,
			param.ErrorMessage,
		)
	}))

	// Настройка доверенных прокси
	router.SetTrustedProxies([]string{"127.0.0.1"})

	// Устанавливаем ограничение на размер загружаемых файлов (20MB)
	router.MaxMultipartMemory = 20 << 20

	// Группа эндпоинтов API
	api := router.Group("/api")
	{
		// Получение списка моделей
		api.GET("/models", getModelsHandler)

		// Транскрипция аудиофайла
		api.POST("/transcribe", transcribeHandler)

		// Информация о статусе сервера
		api.GET("/health", healthCheckHandler)

		// Метрики сервиса
		api.GET("/metrics", metricsHandler)
	}

	// Статические файлы для веб-интерфейса
	router.Static("/static", "./static")
	router.StaticFile("/", "./static/index.html")

	// Запуск сервера
	log.Printf("🚀 Запуск сервера на порту %s в режиме %s...", serverPort, ginMode)
	if err := router.Run(":" + serverPort); err != nil {
		log.Fatalf("❌ Ошибка при запуске сервера: %v", err)
	}
}

// Обработчик для проверки здоровья сервиса
func healthCheckHandler(c *gin.Context) {
	c.JSON(http.StatusOK, gin.H{
		"status":      "ok",
		"server_time": time.Now().Format(time.RFC3339),
		"version":     "1.0.0",
		"uptime":      time.Since(startTime).String(),
		"mode":        ginMode,
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

// Обработчик для метрик сервиса
func metricsHandler(c *gin.Context) {
	metrics := whisperClient.GetMetrics()
	c.JSON(http.StatusOK, gin.H{
		"requests_total":     metrics.RequestsTotal,
		"errors_total":       metrics.ErrorsTotal,
		"processing_time_ms": metrics.ProcessingTimeMs,
		"uptime":             time.Since(startTime).String(),
	})
}

// Обработчик для транскрипции аудио
func transcribeHandler(c *gin.Context) {
	// Обработка файла
	file, err := c.FormFile("file")
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Ошибка при получении файла: " + err.Error()})
		return
	}

	// Проверка размера файла (максимум 100 МБ)
	if file.Size > 100*1024*1024 {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Файл слишком большой. Максимальный размер: 100 МБ"})
		return
	}

	// Создаем временный файл
	tempFile, err := os.CreateTemp("", "whisper-upload-*.tmp")
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Ошибка при создании временного файла: " + err.Error()})
		return
	}
	defer os.Remove(tempFile.Name())
	defer tempFile.Close()

	// Сохраняем загруженный файл
	if err := c.SaveUploadedFile(file, tempFile.Name()); err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Ошибка при сохранении файла: " + err.Error()})
		return
	}

	log.Printf("Начало обработки файла %s (%.2f МБ)", file.Filename, float64(file.Size)/1024/1024)

	// Получаем параметры из запроса
	model := c.DefaultPostForm("model", "base")
	var language *string
	if langValue := c.PostForm("language"); langValue != "" {
		language = &langValue
	}
	task := c.DefaultPostForm("task", "transcribe")

	// Выполняем транскрипцию
	startTime := time.Now()
	result, err := whisperClient.Transcribe(tempFile.Name(), model, language, task)
	elapsedTime := time.Since(startTime)

	// Проверяем ошибки
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{
			"error": fmt.Sprintf("Ошибка при выполнении транскрипции: %v", err),
		})
		return
	}

	// Возвращаем результат с дополнительной информацией о времени обработки
	c.JSON(http.StatusOK, gin.H{
		"status": "success",
		"data":   result,
		"metrics": gin.H{
			"processing_time": elapsedTime.String(),
			"file_size":       file.Size,
			"model":           model,
			"language":        language,
			"task":            task,
		},
	})
}

// Глобальная переменная для отслеживания времени запуска
var startTime = time.Now()
