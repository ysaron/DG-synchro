import numpy as np
import random
from collections import namedtuple, deque
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import FixedLocator
from datetime import datetime
from tqdm import tqdm
import yaml
import sys

matplotlib.use(backend='Qt5Agg')

Event = namedtuple('Event', ['delta_f', 'end_t'])
Deflection = namedtuple('Deflection', ['delta_f', 'end_t'])


class Config:
    """ Оболочка файла конфигурации с контролем корректности введенных значений """

    try:
        with open('config.yaml') as f:
            cfg = yaml.safe_load(f)

        assert type(cfg['f_jump_max']) == float, 'Значение скачка частоты должно быть числом с плавающей точкой'
        assert type(cfg['f_jump_duration_min']) in [int, float], 'Продолжительность скачка должна быть числом'
        assert type(cfg['f_jump_duration_max']) in [int, float], 'Продолжительность скачка должна быть числом'
        assert cfg['f_jump_duration_min'] > 0, 'Продолжительность скачка должна быть положительным числом'
        assert cfg['f_jump_duration_max'] > 0, 'Продолжительность скачка должна быть положительным числом'
        assert type(cfg['jump_probability']) == float, 'Вероятность должна быть числом с плавающей точкой'
        assert 0 < cfg['jump_probability'] < 1, 'Вероятность должна находиться в диапазоне от 0 до 1'
        assert type(cfg['step']) == float, 'Временной шаг должен быть числом с плавающей точкой'
        assert cfg['step'] <= 0.00002, '0.00002с - максимально возможное значение временного шага'
        assert type(cfg['simulation_time']) in [int, float], 'Время симуляции должно быть числом'
        assert cfg['simulation_time'] > 0, 'Время симуляции должно быть положительным числом'
        assert type(cfg['U_RMS']) in [int, float], 'Напряжение должно быть числом'
        assert type(cfg['phi0_min']) in [int, float], 'Значение фазы должно быть числом'
        assert type(cfg['phi0_max']) in [int, float], 'Значение фазы должно быть числом'
        assert 0 <= cfg['phi0_min'] < 360, 'phi0 должно находиться в диапазоне 0:359 ' \
                                           '(программно переводится в -180:+180)'
        assert 0 <= cfg['phi0_max'] < 360, 'phi0 должно находиться в диапазоне 0:359 ' \
                                           '(программно переводится в -180:+180)'
        assert cfg['phi0_min'] <= cfg['phi0_max'], 'phi0_min не может быть больше phi0_max'
        assert type(cfg['f_max']) in [int, float], 'Значение граничной частоты должно быть числом'
        assert type(cfg['f_min']) in [int, float], 'Значение граничной частоты должно быть числом'
        assert cfg['f_max'] > 50.5, 'Верхняя граница частоты должна быть больше 50.5 Гц'
        assert cfg['f_min'] < 49.5, 'Нижняя граница частоты должна быть меньше 49.5 Гц'
        assert type(cfg['RNG_seed']) == int or cfg['RNG_seed'] is None, 'Порождающий элемент ГСЧ - целое число или null'
        assert type(cfg['control']) == bool, 'control может быть равен true или false'
        assert type(cfg['enable_fill']) == bool, 'enable_fill может быть равен true или false'

        utility_frequency = cfg['utility_frequency']
        f_jump_max = abs(cfg['f_jump_max'])
        frequency_jump_values = np.arange(-cfg['f_jump_max'], cfg['f_jump_max'] + 0.1, 0.1)
        frequency_jump_durations = np.arange(cfg['f_jump_duration_min'], cfg['f_jump_duration_max'], 1.0)
        jump_probability = cfg['jump_probability']
        step = cfg['step']
        sim_time = cfg['simulation_time']
        x_meshgrid = np.arange(0, sim_time, step).round(5)
        amplitude = abs(cfg['U_RMS']) * np.sqrt(2)
        phi0 = random.randint(cfg['phi0_min'], cfg['phi0_max'])
        f_max = cfg['f_max']
        f_min = cfg['f_min']
        seed_value = cfg['RNG_seed']
        f_control = cfg['control']
        enable_fill = cfg['enable_fill']
    except FileNotFoundError:
        print('Файл "config.yaml" в текущей директории не обнаружен')
        input('Enter - закрыть окно\n')
        sys.exit()
    except AssertionError as e:
        print(f'Ошибка конфигурации: {e}')
        input('Enter - закрыть окно\n')
        sys.exit()


class DieselGenerator:
    """ Базовый класс, представляющий генератор """
    x_axis = Config.x_meshgrid

    def __init__(self, ampl: float, freq: float, phase: float):
        """
        :param ampl: амплитуда выходного напряжения (В)
        :param freq: частота в начале симуляции (Гц)
        :param phase: начальная фаза (град.)
        """
        self.amplitude = ampl
        self.frequency = freq
        self.phase_0 = phase * np.pi / 180                  # перевод фазы в радианы
        self.instant = 0.0                                  # мгновенное значение напряжения
        self.is_positive: bool = Config.phi0 > 0            # флаг полярности напряжения в данный момент времени
        self.zero_transition: bool = False                  # флаг перехода минус-плюс в данный момент времени
        self.zero_transition_times = deque([], maxlen=2)    # хранение последнего и пред. моментов перехода минус-плюс
        self.u = []                                         # список мгновенных значений напряжения генератора
        self.section_t: float = 0.0      # локальное время текущей части синусоиды (для расчета после изменения частоты)
        self.frequency_changed: bool = False
        self.previous_f = freq           # значение частоты напряжения ДГА в предыдущий момент времени

    def calc_instant(self, global_time: float):
        """
        Вычисляет мгновенное значение напряжения на данной отметке времени
        :param global_time: текущая временная отметка
        """

        if self.frequency_changed:
            self.phase_0 = 2 * np.pi * self.previous_f * self.section_t + self.phase_0
            self.section_t = 0.0

        # округление, т.к. проблема точности вычисления float иногда приводит к числам в -16 степени вместо 0
        self.instant = self.amplitude * np.sin(2 * np.pi * self.frequency * self.section_t + self.phase_0).round(10)

        self.section_t += Config.step
        self.__catch_zero_transition(global_time)
        self.u.append(self.instant)

    def __catch_zero_transition(self, t: float):
        """ Устанавливает флаги для отслеживания переходов минус-плюс синусоиды """

        self.zero_transition = (not self.is_positive) and self.instant >= 0
        self.is_positive = self.instant >= 0

        # Сохранение времени перехода через 0
        if self.zero_transition:
            if len(self.zero_transition_times) == 2:
                self.zero_transition_times.popleft()
            self.zero_transition_times.append(t)


class WorkingDGA(DieselGenerator):
    """ Представляет работающий нагруженный генератор """

    def __init__(self, ampl: float, freq: float, phase: float):
        super().__init__(ampl, freq, phase)
        self.deltas_freq = Config.frequency_jump_values
        self.event_durations = Config.frequency_jump_durations
        self.event_list: list[Event] = []   # список активных скачков частоты
        self.jump: bool = False             # флаг скачкообразного изменения частоты в данный момент времени
        self.calculated_frequency = Config.utility_frequency
        self.f1_rand = []       # значения случайно генерируемой частоты напряжения ДГА1
        self.f1_calc = []       # значения рассчитываемой частоты напряжения ДГА1

    def generate_frequency(self, global_time: float):
        """ Генерация случайно (скачками) изменяющейся частоты """

        self.previous_f = self.frequency    # сохранение предыдущего значения частоты

        # Определение самого факта скачка частоты
        jump = random.choices([True, False], weights=[1, 1 / Config.jump_probability])[0]

        # Определение списков завершающихся и еще активных скачков
        ended_events: list[Event] = [item for item in self.event_list if item.end_t == global_time]
        self.event_list = [item for item in self.event_list if item.end_t != global_time]

        # Завершение скачков
        for item in ended_events:
            self.frequency -= item.delta_f

        if jump:
            # Определение параметров скачка частоты (чем сильнее скачок - тем меньше его вероятность)
            weights = (Config.f_jump_max + 0.1 - abs(self.deltas_freq)) * 10
            delta_f = random.choices(self.deltas_freq, weights=weights)[0]
            event = Event(delta_f=delta_f, end_t=global_time + random.choice(self.event_durations))
            self.event_list.append(event)
            self.frequency += event.delta_f

            # Ограничение частоты
            if self.frequency > Config.f_max:
                self.frequency = Config.f_max
            elif self.frequency < Config.f_min:
                self.frequency = Config.f_min

        # Выявление факта изменения частоты
        if ended_events:
            jump_back = True
        else:
            jump_back = False
        self.jump = jump or jump_back
        self.frequency_changed = True if self.jump else False

        self.f1_rand.append(self.frequency)

    def calculate_frequency(self):
        """ Расчет частоты f1 по переходам через 0 """
        # Частота f1 рассчитывается только в моменты перехода синусоиды минус-плюс
        # иначе - берется равной предыдущему рассчитанному значению
        if all([self.zero_transition,
                len(self.zero_transition_times) == 2]):
            self.calculated_frequency = 1 / (self.zero_transition_times[1] - self.zero_transition_times[0])
            self.calculated_frequency = round(self.calculated_frequency, 1)
        else:
            try:
                self.calculated_frequency = self.f1_calc[-1]
            except IndexError:
                self.calculated_frequency = Config.utility_frequency

        self.f1_calc.append(self.calculated_frequency)


class StartingDGA(DieselGenerator):
    """ Представляет запускаемый генератор, синхронизируемый с работающим """

    def __init__(self, ampl: float, freq: float, phase: float):
        super().__init__(ampl, freq, phase)
        self.f2_calc = []               # значения частоты напряжения ДГА2
        self.d_phi = - Config.phi0
        self.phase_shifts = []          # сдвиги фаз ДГА2 относительно ДГА1
        self.deflection = Deflection(delta_f=0, end_t=0)    # отклонения частоты ДГА2 с целью регулировки сдвига
        self.synchronized: bool = False     # флаг состояния системы
        self.synchro_time = Config.sim_time

    def calculate_deflection(self, global_t: float):
        """ Расчет отклонения частоты f2 для достижения синхронизации """
        k = 0.03    # экспериментально полученный коэффициент для расчета времени отклонения f2
        if 0 <= self.d_phi <= 180:
            delta_f = 0.1
            end_t = self.d_phi * k + global_t
        else:
            delta_f = -0.1
            end_t = -self.d_phi * k + global_t
        self.deflection = Deflection(delta_f=delta_f, end_t=round(end_t, 2))
        tqdm.write(
            f'\nРегулирование сдвига: {round(self.d_phi, 1)}°\t|\tВводится отклонение f2: {self.deflection.delta_f} Гц'
            f'; до момента времени: {self.deflection.end_t} с\n{"-" * 100}')

    def calculate_frequency(self, w_dga: WorkingDGA, current_time: float):
        """ Рассчитывает частоту f2 в данный момент времени """

        self.previous_f = self.frequency  # сохранение предыдущего значения частоты

        if current_time == self.deflection.end_t:
            self.deflection = Deflection(0, self.deflection.end_t)  # обнуление отклонения, f2 --> f1

        if all([not self.synchronized,
                current_time >= self.deflection.end_t + 0.1,
                self.deflection.delta_f == 0,
                Config.f_control]):
            # если синхронизация не наступила и регулировки в данный момент нет - снова регулируем f2
            self.calculate_deflection(current_time)

        try:
            if self.synchronized:
                # после наступления синхронизации частота f2 изменяется синхронно с f1
                self.frequency = w_dga.frequency
            else:
                self.frequency = w_dga.f1_calc[-50] + self.deflection.delta_f
        except IndexError:
            self.frequency = Config.utility_frequency

        self.frequency_changed = self.previous_f != self.frequency
        self.f2_calc.append(self.frequency)

    def calculate_phase_shift(self, w_dga: WorkingDGA, current_time: float):
        """ Рассчитывает текущий фазовый сдвиг между ДГА1 и ДГА2 """

        if all([self.zero_transition,
                len(self.zero_transition_times) == 2]):
            period = self.zero_transition_times[1] - self.zero_transition_times[0]
            shift = self.zero_transition_times[1] - w_dga.zero_transition_times[1]
            self.d_phi = 2 * np.pi * shift / period
            self.d_phi = self.d_phi * 180 / np.pi
        else:
            try:
                self.d_phi = self.phase_shifts[-1]
            except IndexError:
                self.d_phi = Config.phi0

        self.d_phi = self.convert_angle(self.d_phi)
        self.phase_shifts.append(self.d_phi)
        if all([w_dga.calculated_frequency == self.frequency,
                -0.3 < self.d_phi < 0.3,
                not self.synchronized]):
            self.synchronized = True
            tqdm.write(f'\nСИНХРОНИЗИРОВАНО. Коммутация в {round(current_time, 2)} с')
            self.synchro_time = current_time

    @staticmethod
    def convert_angle(angle: float):
        """ Конвертация угла фазового сдвига из диапазона (0:360°) в (-180:180°) """
        if angle > 180.0:
            return angle - 360
        else:
            return angle


def main():
    start = datetime.now()

    f_01 = random.choice(np.arange(-0.4, 0.5, 0.1)) + Config.utility_frequency      # Значение частоты ДГА1 в момент t0
    dga1 = WorkingDGA(ampl=Config.amplitude, freq=f_01, phase=Config.phi0)
    dga2 = StartingDGA(ampl=Config.amplitude, freq=Config.utility_frequency, phase=0)

    tqdm.write(f'\nВремя симуляции: {Config.sim_time} с')
    tqdm.write(f'Регулирование фазового сдвига: {"да" if Config.f_control else "нет"}')
    tqdm.write(f'φ0 ДГА1 = {StartingDGA.convert_angle(Config.phi0)}°')

    for global_t in tqdm(DieselGenerator.x_axis, desc='Симуляция', ncols=100):
        dga1.generate_frequency(global_t)           # генерация случайной частоты ДГА1
        dga1.calc_instant(global_time=global_t)     # расчет u1
        dga1.calculate_frequency()                  # расчет частоты ДГА1
        dga2.calculate_frequency(dga1, global_t)    # расчет частоты ДГА2
        dga2.calc_instant(global_time=global_t)     # расчет u2
        dga2.calculate_phase_shift(dga1, global_t)  # расчет фазового сдвига между ДГА2 и ДГА1

    if not dga2.synchronized:
        tqdm.write(f'За {Config.sim_time} с синхронизация НЕ наступила.')
    tqdm.write(f'{"_" * 100}\nРасчеты заняли {(datetime.now() - start).seconds} с')

    # Инициализация графиков и координатных сеток
    fig1: Figure = plt.figure(figsize=(7, 4), facecolor='#D3D3D3')
    ax1: Axes = fig1.add_subplot(211)
    ax2: Axes = fig1.add_subplot(212)
    fig2: Figure = plt.figure(figsize=(7, 4), facecolor='#D3D3D3')
    ax3: Axes = fig2.add_subplot(211)
    ax4: Axes = fig2.add_subplot(212)

    # Построение графиков
    ax1.plot(DieselGenerator.x_axis, dga1.f1_calc, label='f1 рассчитанная', color='#B22222')
    ax1.plot(DieselGenerator.x_axis, dga1.f1_rand, '--', label='f1 сгенерированная', alpha=0.5, color='#00008B')
    ax2.plot(DieselGenerator.x_axis, dga1.u, '--', label='u ДГА1', color='#FF4500')
    ax2.plot(DieselGenerator.x_axis, dga2.u, label='u ДГА2', alpha=0.5, color='#32CD32')
    ax3.plot(DieselGenerator.x_axis, dga1.f1_calc, '--', label='f1 рассчитанная', color='#B22222')
    ax3.plot(DieselGenerator.x_axis, dga2.f2_calc, label='f2', color='#32CD32')
    ax3.plot(DieselGenerator.x_axis, dga1.f1_rand, label='f1 сгенерированная', alpha=0.35, color='#00008B')
    ax4.plot(DieselGenerator.x_axis, dga2.phase_shifts, label='Δφ', color='#4682B4')

    # Установка границ
    ax2.set_ylim(-200 * Config.amplitude, 200 * Config.amplitude)
    ax4.set_ylim(-180, 180)
    ax1.set_xlim(0, Config.sim_time)
    ax2.set_xlim(0, Config.sim_time)
    ax3.set_xlim(0, Config.sim_time)
    ax4.set_xlim(0, Config.sim_time)

    # Сетка, легенда, подписи осей
    ax1.grid()
    ax2.grid()
    ax3.grid()
    ax4.grid()
    ax1.legend(facecolor='#D3D3D3', edgecolor='#000000', loc='best')
    ax2.legend(facecolor='#D3D3D3', edgecolor='#000000', loc='best')
    ax3.legend(facecolor='#D3D3D3', edgecolor='#000000', loc='best')
    ax4.legend(facecolor='#D3D3D3', edgecolor='#000000', loc='best')
    ax1.set_xlabel('t, с', size=16)
    ax1.set_ylabel('f, Гц', rotation='horizontal', ha='right', size=16)
    ax2.set_xlabel('t, с', size=16)
    ax2.set_ylabel('u, В', rotation='horizontal', ha='right', size=16)
    ax3.set_xlabel('t, с', size=16)
    ax3.set_ylabel('f, Гц', rotation='horizontal', ha='right', size=16)
    ax4.set_xlabel('t, с', size=16)
    ax4.set_ylabel('Δφ, °', rotation='horizontal', ha='right', size=16)

    # Отрисовка линии в момент синхронизации
    ax1.axvline(x=dga2.synchro_time, color='#000000', alpha=0.3, linestyle='--')
    ax2.axvline(x=dga2.synchro_time, color='#000000', alpha=0.3, linestyle='--')
    ax3.axvline(x=dga2.synchro_time, color='#000000', alpha=0.3, linestyle='--')
    ax4.axvline(x=dga2.synchro_time, color='#000000', alpha=0.3, linestyle='--')

    if Config.enable_fill:
        # Заливка областей до и после синхронизации
        ax1.axvspan(xmin=0, xmax=dga2.synchro_time, color='#8B0000', alpha=0.1)
        ax1.axvspan(xmin=dga2.synchro_time, xmax=Config.sim_time, color='#006400', alpha=0.1)
        ax2.axvspan(xmin=0, xmax=dga2.synchro_time, color='#8B0000', alpha=0.1)
        ax2.axvspan(xmin=dga2.synchro_time, xmax=Config.sim_time, color='#006400', alpha=0.1)
        ax3.axvspan(xmin=0, xmax=dga2.synchro_time, color='#8B0000', alpha=0.1)
        ax3.axvspan(xmin=dga2.synchro_time, xmax=Config.sim_time, color='#006400', alpha=0.1)
        ax4.axvspan(xmin=0, xmax=dga2.synchro_time, color='#8B0000', alpha=0.1)
        ax4.axvspan(xmin=dga2.synchro_time, xmax=Config.sim_time, color='#006400', alpha=0.1)

    ax3.fill_between(DieselGenerator.x_axis, dga2.f2_calc, dga1.f1_calc,
                     color='#DCDCDC', alpha=0.65, where=(DieselGenerator.x_axis < dga2.synchro_time))

    # Произвольный текст на графике
    ax2.text(-2.5, -Config.amplitude * 50, f'φ0 ДГА1 = {StartingDGA.convert_angle(Config.phi0)}°')
    ax4.text(-2.5, -30, f'φ0 ДГА1 = {StartingDGA.convert_angle(Config.phi0)}°')

    # Настройка меток осей
    ax2.yaxis.set_major_locator(FixedLocator([-Config.amplitude, 0, Config.amplitude]))

    plt.show()


if __name__ == '__main__':
    random.seed(Config.seed_value)
    main()
