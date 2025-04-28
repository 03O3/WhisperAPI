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
import queue
import concurrent.futures
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
DEFAULT_WORKERS = max(1, os.cpu_count() - 1)  # Количество рабочих потоков по умолчанию

# Переменная для хранения моделей
models = {}
model_lock = threading.Lock()  # Блокировка для доступа к моделям

# Пул потоков для обработки задач транскрипции
task_queue = queue.Queue()
results = {}  # Словарь для хранения результатов по task_id
results_lock = threading.Lock()  # Блокировка для доступа к результатам

def get_model(model_name="base"):
    """Получить модель из кэша или загрузить новую"""
    with model_lock:
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

def worker_task():
    """Функция для рабочего потока, обрабатывающего задачи из очереди"""
    while True:
        try:
            # Получаем задачу из очереди
            task_id, audio_path, options = task_queue.get()
            
            try:
                # Выполняем транскрипцию
                result = transcribe_audio(
                    audio_path,
                    model_name=options.get('model', 'base'),
                    language=options.get('language'),
                    task=options.get('task', 'transcribe')
                )
                
                # Сохраняем результат
                with results_lock:
                    results[task_id] = {
                        "status": "completed",
                        "result": result
                    }
                
                # Удаляем временные файлы, если они были созданы
                if options.get('temp_file') and os.path.exists(audio_path):
                    try:
                        os.remove(audio_path)
                        temp_dir = os.path.dirname(audio_path)
                        if os.path.exists(temp_dir):
                            os.rmdir(temp_dir)
                    except Exception as e:
                        logger.warning(f"Не удалось удалить временный файл: {str(e)}")
                
            except Exception as e:
                logger.error(f"Ошибка в рабочем потоке при обработке задачи {task_id}: {str(e)}")
                with results_lock:
                    results[task_id] = {
                        "status": "error",
                        "error": str(e)
                    }
            
            finally:
                # Отмечаем задачу как выполненную
                task_queue.task_done()
        
        except Exception as e:
            logger.error(f"Ошибка в рабочем потоке: {str(e)}")

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
                        # Генерируем уникальный ID задачи
                        task_id = str(time.time()) + "_" + str(threading.get_ident())
                        
                        # Сохраняем временный файл
                        temp_dir = None
                        audio_path = None
                        is_temp_file = False
                        
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
                                # Добавляем задачу в очередь
                                logger.info(f"Файл найден, добавляем задачу в очередь: {audio_path}")
                                
                                # Проверяем, нужно ли ожидать результат
                                if message.get('async', False):
                                    task_queue.put((
                                        task_id,
                                        audio_path,
                                        {
                                            'model': message.get('model', 'base'),
                                            'language': message.get('language'),
                                            'task': message.get('task', 'transcribe'),
                                            'temp_file': False
                                        }
                                    ))
                                    
                                    response = {
                                        "status": "accepted",
                                        "task_id": task_id,
                                        "message": "Задача добавлена в очередь"
                                    }
                                else:
                                    # Синхронное выполнение (старое поведение)
                                    response = transcribe_audio(
                                        audio_path,
                                        model_name=message.get('model', 'base'),
                                        language=message.get('language'),
                                        task=message.get('task', 'transcribe')
                                    )
                        else:
                            # Сохраняем данные во временный файл
                            temp_dir = tempfile.mkdtemp()
                            temp_file = os.path.join(temp_dir, "temp_audio.mp3")
                            try:
                                # Декодируем бинарные данные из base64
                                binary_data = base64.b64decode(audio_data)
                                with open(temp_file, 'wb') as f:
                                    f.write(binary_data)
                                
                                audio_path = temp_file
                                is_temp_file = True
                                
                                logger.info(f"Данные получены напрямую, сохранены во временный файл: {temp_file}")
                                
                                # Проверяем, нужно ли ожидать результат
                                if message.get('async', False):
                                    task_queue.put((
                                        task_id,
                                        audio_path,
                                        {
                                            'model': message.get('model', 'base'),
                                            'language': message.get('language'),
                                            'task': message.get('task', 'transcribe'),
                                            'temp_file': True
                                        }
                                    ))
                                    
                                    response = {
                                        "status": "accepted",
                                        "task_id": task_id,
                                        "message": "Задача добавлена в очередь"
                                    }
                                else:
                                    # Синхронное выполнение (старое поведение)
                                    response = transcribe_audio(
                                        audio_path,
                                        model_name=message.get('model', 'base'),
                                        language=message.get('language'),
                                        task=message.get('task', 'transcribe')
                                    )
                                    
                                    # Удаляем временные файлы в синхронном режиме
                                    try:
                                        if os.path.exists(temp_file):
                                            os.remove(temp_file)
                                        if temp_dir and os.path.exists(temp_dir):
                                            os.rmdir(temp_dir)
                                    except:
                                        pass
                            except Exception as e:
                                logger.error(f"Ошибка при обработке аудиоданных: {str(e)}")
                                response = {"error": f"Ошибка при обработке аудиоданных: {str(e)}"}
                                # Удаляем временные файлы в случае ошибки
                                try:
                                    if os.path.exists(temp_file):
                                        os.remove(temp_file)
                                    if temp_dir and os.path.exists(temp_dir):
                                        os.rmdir(temp_dir)
                                except:
                                    pass
                    
                    elif message['command'] == 'get_result':
                        # Получаем результат по ID задачи
                        task_id = message.get('task_id')
                        
                        if not task_id:
                            response = {"error": "Не указан ID задачи"}
                        else:
                            with results_lock:
                                result = results.get(task_id)
                                
                                if result:
                                    response = result
                                    # Если задача завершена, удаляем её результат из словаря
                                    if result.get('status') in ['completed', 'error']:
                                        del results[task_id]
                                else:
                                    response = {
                                        "status": "pending",
                                        "message": "Задача ещё не завершена или ID задачи не найден"
                                    }
                    
                    elif message['command'] == 'list_models':
                        available_models = {
                            "tiny": "Самая быстрая модель (~1 ГБ)",
                            "base": "Хороший баланс скорости и точности (~1 ГБ)",
                            "small": "Более точная модель (~2 ГБ)",
                            "medium": "Еще более точная модель (~5 ГБ)",
                            "large": "Самая точная модель (~10 ГБ)"
                        }
                        
                        with model_lock:
                            loaded_models = list(models.keys())
                        
                        response = {
                            "available_models": available_models,
                            "loaded_models": loaded_models
                        }
                    
                    elif message['command'] == 'queue_status':
                        # Информация о состоянии очереди задач
                        response = {
                            "queue_size": task_queue.qsize(),
                            "active_tasks": len(results)
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

def start_worker_threads(num_workers):
    """Запуск рабочих потоков для обработки задач"""
    for _ in range(num_workers):
        worker = threading.Thread(target=worker_task)
        worker.daemon = True
        worker.start()
        logger.info(f"Запущен рабочий поток #{_ + 1}")

def start_server(host=DEFAULT_HOST, port=DEFAULT_PORT, num_workers=DEFAULT_WORKERS):
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
        # Запускаем рабочие потоки для обработки задач
        start_worker_threads(num_workers)
        
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((host, port))
        server.listen(5)
        
        # Устанавливаем таймаут для socket.accept(), чтобы периодически проверять флаг server_running
        server.settimeout(1.0)
        
        logger.info(f"Сервер запущен на {host}:{port} с {num_workers} рабочими потоками (нажмите Ctrl+C для завершения)")
        
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
    num_workers = int(os.environ.get('WHISPER_WORKERS', DEFAULT_WORKERS))
    
    # Запускаем сервер
    start_server(host, port, num_workers) 