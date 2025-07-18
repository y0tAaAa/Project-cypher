import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
import torch.multiprocessing as mp
from enhanced_model import ModelConfig, EnhancedCipherModel
import pandas as pd
from transformers import set_seed
import wandb
from datetime import datetime

def setup(rank, world_size):
    """Настройка распределенного обучения"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    """Очистка после обучения"""
    dist.destroy_process_group()

def train_model(rank, world_size, config):
    """Функция обучения для одного GPU"""
    setup(rank, world_size)
    
    # Устанавливаем seed для воспроизводимости
    set_seed(42 + rank)
    
    # Инициализируем wandb только для основного процесса
    if rank == 0:
        run_name = f"cipher_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        wandb.init(
            project="enhanced-cipher",
            name=run_name,
            config={
                "model_name": config.base_model_name,
                "batch_size": config.training_args.per_device_train_batch_size,
                "learning_rate": config.training_args.learning_rate,
                "epochs": config.training_args.num_train_epochs,
            }
        )
    
    # Модифицируем конфигурацию для распределенного обучения
    config.training_args.local_rank = rank
    config.device_map = f"cuda:{rank}"
    
    # Инициализируем модель
    model = EnhancedCipherModel(config)
    model.load_base_model()
    
    # Загружаем данные
    train_data = pd.read_csv("data/train_enhanced_vigenere.csv")
    eval_data = pd.read_csv("data/val_enhanced_vigenere.csv")
    
    # Добавляем данные для substitution cipher
    train_data_sub = pd.read_csv("data/train_enhanced_substitution.csv")
    eval_data_sub = pd.read_csv("data/val_enhanced_substitution.csv")
    
    train_data = pd.concat([train_data, train_data_sub], ignore_index=True)
    eval_data = pd.concat([eval_data, eval_data_sub], ignore_index=True)
    
    # Создаем датасеты
    train_dataset = model.prepare_dataset(train_data)
    eval_dataset = model.prepare_dataset(eval_data)
    
    # Настраиваем распределенную выборку
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )
    
    # Оборачиваем модель в DistributedDataParallel
    model.model = DistributedDataParallel(
        model.model,
        device_ids=[rank],
        output_device=rank,
        find_unused_parameters=True
    )
    
    # Обновляем параметры обучения
    model.config.training_args.dataloader_num_workers = 4
    model.config.training_args.dataloader_pin_memory = True
    
    # Запускаем обучение
    try:
        model.train(train_data, eval_data)
        
        # Сохраняем модель только в основном процессе
        if rank == 0:
            model.save_model(f"results/enhanced_cipher_model_final_{run_name}")
            wandb.finish()
    
    except Exception as e:
        print(f"Error in rank {rank}: {str(e)}")
        raise e
    finally:
        cleanup()

def main():
    """Основная функция запуска"""
    # Конфигурация для 4x NVIDIA 1080
    config = ModelConfig(
        base_model_name="EleutherAI/gpt-neo-2.7B",  # Меньше чем 20B, но все еще мощная
        model_max_length=1024,
        gradient_checkpointing=True,
        torch_dtype=torch.float16,  # Используем fp16 вместо bfloat16 для старых карт
        load_in_8bit=True,
    )
    
    # Обновляем параметры обучения для распределенного режима
    config.training_args.per_device_train_batch_size = 2
    config.training_args.gradient_accumulation_steps = 8
    config.training_args.learning_rate = 1e-4
    config.training_args.warmup_steps = 200
    config.training_args.max_grad_norm = 1.0
    config.training_args.logging_steps = 10
    config.training_args.save_strategy = "steps"
    config.training_args.save_steps = 500
    config.training_args.eval_steps = 500
    config.training_args.num_train_epochs = 3
    
    # Определяем количество доступных GPU
    world_size = torch.cuda.device_count()
    print(f"Starting distributed training on {world_size} GPUs...")
    
    # Запускаем распределенное обучение
    mp.spawn(
        train_model,
        args=(world_size, config),
        nprocs=world_size,
        join=True
    )

if __name__ == "__main__":
    main() 