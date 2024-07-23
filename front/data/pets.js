const pets = [
  {
    id: 1,
    name: 'Импорт библиотек и загрузка данных',
    img: 'assets/images/tigers/1.jpg',
    description:
      "Импортируются необходимые библиотеки, такие как PyTorch, torchvision, matplotlib, OpenCV и другие. Загружаются данные из директории tigers_dataset с помощью библиотеки pathlib.",
  },
  {
    id: 2,
    name: 'Определение класса датасета',
    img: 'assets/images/tigers/2.jpg',
    description:
      "Определяется класс TigerDataset для работы с данными. Класс наследуется от torch.utils.data.Dataset и переопределяет методы init, len и getitem.",
  },
  {
    id: 3,
    name: 'Инициализация датасетов',
    img: 'assets/images/tigers/3.jpg',
    description:
      'Инициализируются тренировочные, валидационные и тестовые датасеты с помощью класса TigerDataset. Создаются данные для загрузки с помощью класса DataLoader из PyTorch.',
  },
  {
    id: 4,
    name: 'Определение модели сети',
    img: 'assets/images/tigers/4.jpg',
    description:
      'Определяется модель сети UNet с помощью класса nn.Module из PyTorch.',
  },
  {
    id: 5,
    name: 'Определение функции потерь',
    img: 'assets/images/tigers/5.jpg',
    description:
      'Определяется функция потерь dice_loss и класс DiceBCELoss для расчета потерь.',
  },
  {
    id: 6,
    name: 'Определение оптимизатора',
    img: 'assets/images/tigers/6.jpg',
    description:
      'Определяется оптимизатор Adam из PyTorch для обучения модели.',
  },
  {
    id: 7,
    name: 'Обучается модель с помощью функции',
    img: 'assets/images/tigers/7.jpg',
    description:
      'Обучается модель с помощью функции train и evaluate. Функция train обучает модель на тренировочном датасете, а функция evaluate оценивает модель на валидационном датасете.',
  },
  {
    id: 8,
    name: 'Вывод результатов и сохранение модели',
    img: 'assets/images/tigers/8.jpg',
    description:
      'Выводятся результаты обучения модели, включая потери и точность, каждые 2 эпохи. Модель сохраняется в файл best_model_v1.pt при достижении лучшей потери на валидационном датасете.',

  },
];

export default pets;
