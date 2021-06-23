# DG-synchro
Скрипт моделирует ускоренную точную синхронизацию двух дизель-генераторов (ДГА) за счет регулирования частоты вращения генератора, а следовательно, и частоты выходного напряжения.
____
### Обзор
Моделируется 2 генератора:
- Находящийся в работе, нагруженный (ДГА1). Нагрузка произвольным образом изменяется с течением времени (включение/отключение электроприборов), что приводит к ступенчатому изменению частоты выходного напряжения.
- Пускаемый в начальный момент времени (ДГА2). Частота его напряжения регулируется для выполнения условий точной синхронизации.
____
### Настройка
В конфигурационном файле задаются:
- диапазоны значений параметров скачков частоты ДГА1;
- вероятность скачков;
- диапазон значений фазы ДГА1 в начальный момент времени;
- ограничения значений частоты;
- прочие параметры.
____
### Иллюстрации

![console](/Pics/1.jpg)

Масштабированный график изменения мгновенных напряжений ДГА1 и ДГА2 в начале симуляции, в момент синхронизации и в конце симуляции:

![u](/Pics/2.jpg)

Изменение частот и фазового сдвига во времени:

![f_phi](/Pics/3.jpg)

____
____
