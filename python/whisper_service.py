import os
import sys
import platform
import tempfile
import json
import time
import socket
import threading
import logging
import base64
import signal
import warnings
from pathlib import Path

# Подавляем предупреждение о FP16
warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead")

# Настраиваем логирование
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("whisper_service")

# Проверка версии Python
python_version = platform.python_version_tuple()
if int(python_version[0]) == 3 and int(python_version[1]) >= 12:
    logger.warning(f"ВНИМАНИЕ: Вы используете Python {platform.python_version()}")
    logger.warning("Whisper может быть несовместим с Python версии 3.12 и выше.")
    logger.warning("Рекомендуется использовать Python версии 3.8-3.11.")

try:
    import whisper
except ImportError as e:
    logger.error(f"Ошибка импорта библиотеки whisper: {e}")
    logger.error("\nПопробуйте выполнить следующие команды для устранения проблемы:")
    logger.error("pip install --upgrade pip")
    logger.error("pip install setuptools wheel")
    logger.error("pip install git+https://github.com/openai/whisper.git")
    sys.exit(1)

# Константы для сетевого взаимодействия
DEFAULT_HOST = '127.0.0.1'
DEFAULT_PORT = 9000
BUFFER_SIZE = 4096
HEADER_SIZE = 8  # Размер заголовка для передачи длины сообщения

# Переменная для хранения моделей
models = {}

def get_model(model_name="base"):
    """Получить модель из кэша или загрузить новую"""
    if model_name not in models:
        logger.info(f"Загрузка модели '{model_name}'...")
        models[model_name] = whisper.load_model(model_name)
    return models[model_name]

def transcribe_audio(audio_path, model_name="base", language=None, task="transcribe"):
    """Распознавание речи в аудиофайле"""
    try:
        # Загрузка модели
        model = get_model(model_name)
        
        # Опции распознавания
        options = {"task": task}
        if language:
            options["language"] = language
        
        # Распознавание
        logger.info(f"Начало распознавания файла {audio_path}")
        start_time = time.time()
        result = model.transcribe(audio_path, **options)
        elapsed_time = time.time() - start_time
        
        # Формирование ответа
        response = {
            "text": result["text"],
            "language": result["language"],
            "segments": result["segments"],
            "processing_time": elapsed_time
        }
        
        logger.info(f"Распознавание завершено за {elapsed_time:.2f} сек")
        return response
    
    except Exception as e:
        logger.error(f"Ошибка при распознавании: {str(e)}")
        return {"error": str(e)}

def handle_client(client_socket):
    """Обработка соединения с клиентом"""
    data_buffer = b''
    message_length = None
    
    try:
        while True:
            # Получаем данные от клиента
            chunk = client_socket.recv(BUFFER_SIZE)
            if not chunk:
                break
            
            data_buffer += chunk
            
            # Если длина сообщения еще не определена и буфер достаточно большой
            if message_length is None and len(data_buffer) >= HEADER_SIZE:
                # Получаем длину сообщения из первых HEADER_SIZE байт
                message_length = int.from_bytes(data_buffer[:HEADER_SIZE], byteorder='big')
                data_buffer = data_buffer[HEADER_SIZE:]
            
            # Если получили полное сообщение
            if message_length is not None and len(data_buffer) >= message_length:
                message_data = data_buffer[:message_length]
                data_buffer = data_buffer[message_length:]
                
                # Обрабатываем сообщение
                try:
                    message = json.loads(message_data.decode('utf-8'))
                    logger.info(f"Получен запрос: {message['command']}")
                    
                    if message['command'] == 'transcribe':
                        # Сохраняем временный файл
                        temp_dir = tempfile.mkdtemp()
                        audio_data = message.get('audio_data')
                        
                        if not audio_data:
                            # Если данные не переданы напрямую, используем путь
                            audio_path = message.get('audio_path')
                            logger.info(f"Получен путь к файлу: {audio_path}")
                            
                            if not audio_path:
                                logger.error("Путь к аудиофайлу не указан")
                                response = {"error": "Путь к аудиофайлу не указан"}
                            elif not os.path.exists(audio_path):
                                logger.error(f"Файл не найден по пути: {audio_path}")
                                response = {"error": "Аудиофайл не найден"}
                            else:
                                logger.info(f"Файл найден, начинаем транскрипцию: {audio_path}")
                                response = transcribe_audio(
                                    audio_path,
                                    model_name=message.get('model', 'base'),
                                    language=message.get('language'),
                                    task=message.get('task', 'transcribe')
                                )
                        else:
                            # Сохраняем данные во временный файл
                            temp_file = os.path.join(temp_dir, "temp_audio.mp3")
                            try:
                                # Декодируем бинарные данные из base64
                                binary_data = base64.b64decode(audio_data)
                                with open(temp_file, 'wb') as f:
                                    f.write(binary_data)
                                    
                                logger.info(f"Данные получены напрямую, сохранены во временный файл: {temp_file}")
                                response = transcribe_audio(
                                    temp_file,
                                    model_name=message.get('model', 'base'),
                                    language=message.get('language'),
                                    task=message.get('task', 'transcribe')
                                )
                            except Exception as e:
                                logger.error(f"Ошибка при обработке аудиоданных: {str(e)}")
                                response = {"error": f"Ошибка при обработке аудиоданных: {str(e)}"}
                            finally:
                                # Удаляем временные файлы
                                try:
                                    os.remove(temp_file)
                                    os.rmdir(temp_dir)
                                except:
                                    pass
                    
                    elif message['command'] == 'list_models':
                        available_models = {
                            "tiny": "Самая быстрая модель (~1 ГБ)",
                            "base": "Хороший баланс скорости и точности (~1 ГБ)",
                            "small": "Более точная модель (~2 ГБ)",
                            "medium": "Еще более точная модель (~5 ГБ)",
                            "large": "Самая точная модель (~10 ГБ)"
                        }
                        
                        loaded_models = list(models.keys())
                        
                        response = {
                            "available_models": available_models,
                            "loaded_models": loaded_models
                        }
                    
                    else:
                        response = {"error": f"Неизвестная команда: {message['command']}"}
                
                except json.JSONDecodeError:
                    response = {"error": "Некорректный формат JSON"}
                except Exception as e:
                    response = {"error": f"Ошибка при обработке запроса: {str(e)}"}
                
                # Отправляем ответ
                response_data = json.dumps(response).encode('utf-8')
                response_length = len(response_data)
                
                # Отправляем сначала длину сообщения, затем само сообщение
                client_socket.sendall(response_length.to_bytes(HEADER_SIZE, byteorder='big'))
                client_socket.sendall(response_data)
                
                # Сбрасываем для следующего сообщения
                message_length = None
    
    except ConnectionError:
        logger.warning("Соединение с клиентом прервано")
    finally:
        client_socket.close()

def start_server(host=DEFAULT_HOST, port=DEFAULT_PORT):
    """Запуск сервера"""
    # Флаг для индикации состояния сервера
    server_running = True
    
    # Обработчик сигнала SIGINT (Ctrl+C)
    def signal_handler(sig, frame):
        nonlocal server_running
        logger.info("Получен сигнал прерывания, завершение работы...")
        server_running = False
        if 'server' in locals():
            server.close()
        sys.exit(0)
    
    # Регистрируем обработчик сигнала
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((host, port))
        server.listen(5)
        
        # Устанавливаем таймаут для socket.accept(), чтобы периодически проверять флаг server_running
        server.settimeout(1.0)
        
        logger.info(f"Сервер запущен на {host}:{port} (нажмите Ctrl+C для завершения)")
        
        # Список активных потоков клиентов
        active_threads = []
        
        while server_running:
            try:
                client_sock, address = server.accept()
                logger.info(f"Новое соединение с {address[0]}:{address[1]}")
                
                client_thread = threading.Thread(target=handle_client, args=(client_sock,))
                client_thread.daemon = True
                client_thread.start()
                active_threads.append(client_thread)
                
                # Очищаем список от завершенных потоков
                active_threads = [t for t in active_threads if t.is_alive()]
            except socket.timeout:
                # Таймаут socket.accept() - просто продолжаем цикл
                continue
            except KeyboardInterrupt:
                # Дополнительная обработка прерывания в цикле
                server_running = False
                break
    
    except Exception as e:
        logger.error(f"Ошибка при запуске сервера: {str(e)}")
    finally:
        logger.info("Сервер остановлен")
        if 'server' in locals():
            server.close()

if __name__ == '__main__':
    # Получение параметров из аргументов командной строки или переменных окружения
    host = os.environ.get('WHISPER_HOST', DEFAULT_HOST)
    port = int(os.environ.get('WHISPER_PORT', DEFAULT_PORT))
    
    # Запускаем сервер
    start_server(host, port) 