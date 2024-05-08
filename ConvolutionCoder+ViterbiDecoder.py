import matplotlib.pyplot as plt
import numpy as np

# Функция для вычисления ветвевой метрики
def branch_metric(encoded_bits, state, bit):
    expected_bits = output_table[state][bit]    
    return -sum(encoded_bits * expected_bits)

mod_ord = 2  # Порядок модуляции QPSK, т.е. число бит на символ сигнального созвездия
n_fft = 64   # Размер БПФ 
n_used = 52
n_symb = 20 * mod_ord # Число символов OFDM
n_cp = 5  # Размер циклического префикс (в сэмплах)

seed_rng = 13  # Затравка для системного генератора псевдослучайных чисел
rng = np.random.default_rng(seed_rng)

fix_seed_rng = 3 # Затравка для "фиксированного" генератора псевдослучайных чисел
fix_rng = np.random.default_rng(fix_seed_rng)

input_bits = rng.integers(0, 2, n_used * n_symb, dtype=np.uint8) # Последовательность случайно сгенерированных бит

fix_bits = fix_rng.integers(0, 2, mod_ord * n_used) # Последовательность "фиксированных" случайно сгенерированных бит

m = 7  # Длина ограничения сверточного кодера

# Инициализация кодера
shift_register = np.zeros(m, dtype=np.uint8)  # Регистр сдвига для сверточного кодера
output_bits = np.zeros(2 * n_used * n_symb, dtype=np.uint8)  # Выходная битовая последовательность от кодера

# Процесс сверточного кодирования
for t in range(len(input_bits)):
  # Сдвигаем регистр вправо и вводим новый бит
  shift_register[1:] = shift_register[:-1]
  shift_register[0] = input_bits[t]

  # Вычисляем два выходных бита на основе генераторных полиномов
  output_bits[2 * t] = shift_register[0] ^ shift_register[2] ^ shift_register[3] ^ shift_register[5] ^ shift_register[6]
  output_bits[2 * t + 1] = shift_register[0] ^ shift_register[1] ^ shift_register[2] ^ shift_register[3] ^ shift_register[6]
  
# QPSK-модулированный сигнал
sig_qpsk = (1.0 - 2.0 * output_bits[::mod_ord]) + 1j * (1.0 - 2.0 * output_bits[1::mod_ord])
sig_qpsk *= np.sqrt(0.5)  # Нормализация QPSK сигнала, чтобы сделать его единичным по мощности

# Столбец QPSK-модулированного "фиксированного" сигнала 
fix_sig_qpsk = (1.0 - 2.0 * fix_bits[::mod_ord]) + 1j * (1.0 - 2.0 * fix_bits[1::mod_ord])
fix_sig_qpsk *= np.sqrt(0.5)

# Изменение размерностей сигнального массива: двумерный массив с размерами N_SYMB x N_FFT
sig_used = np.reshape(sig_qpsk, (n_symb, n_used))

sig_mapped = np.zeros(n_symb * n_fft, dtype=complex)
sig_mapped = np.reshape(sig_mapped, (n_symb, n_fft))

# Входы с 27 по 37 и вход 0 установлены на ноль
sig_mapped[:, 1:27] = sig_used[:, 26:]
sig_mapped[:, 38:] = sig_used[:, 0:26]

# Для фиксированного столбца
fix_sig_qpsk_mapped = np.zeros(n_fft, dtype=complex)
fix_sig_qpsk_mapped[1:27] = fix_sig_qpsk[26:]
fix_sig_qpsk_mapped[38:] = fix_sig_qpsk[0:26]

fix_sig_mapped = np.reshape(fix_sig_qpsk_mapped, (1, n_fft))

# Добавление "фиксированного" столбца в общий массив
sig_mapped_full = np.concatenate((fix_sig_mapped, sig_mapped), axis = 0)

sig_ifft = np.fft.ifft(sig_mapped_full, n=n_fft, norm='ortho')

# Add cyclic prefix (CP). Добавление циклического префикса в начало каждого OFDM сивола
sig_cp = np.concatenate((sig_ifft[:, -n_cp::], sig_ifft), axis = -1)  # 2D arrays concatenation along the last axis

# Channel TDL-D   
delta_f = 15e3 # 0,015 Hz
T_cp = n_cp/(n_fft * delta_f)
delta_t = 0.125 * T_cp # Delay spread, sec
normalized_delay = np.array([0, 0, 0.035, 0.612, 1.363, 1.405, 1.804, 2.596, 1.775, 4.042, 7.937, 9.424, 9.708, 12.525])
p_dB = np.array([-0.2, -13.5, -18.8, -21, -22.8, -17.9, -20.1, -21.9, -22.9, -27.8, -23.6, -24.8, -30, -27.7])
p = 10 ** (p_dB * 0.1)
p = p / p.sum()
tau = delta_t * normalized_delay
fi = rng.uniform(0, 2 * np.pi, 1) # Равномерное распределение
A = rng.standard_normal(len(p)) + 1j * rng.standard_normal(len(p))
A *= np.sqrt(p)
A *= np.sqrt(0.5)
A[0] = np.sqrt(p[0]) * np.exp(1j * fi) 
tau_sampled = np.floor(tau * (n_fft * delta_f))
cir = np.zeros(np.intp(np.max(tau_sampled))+1, dtype=complex)
for tau_value in np.unique(tau_sampled):
    cir[np.intp(tau_value)] = np.sum(A[tau_value == tau_sampled])
    
sig_cp = np.ravel(sig_cp)
sig_cp_conv = np.convolve(sig_cp, cir)
sig_cp_conv = np.reshape(sig_cp_conv[:(n_fft + n_cp) * (n_symb + 1)], (n_symb + 1, n_fft + n_cp))

# Количество пакетов
number_of_tests = 50
# Диапазон отношения сигнал-шум в дБ
snr = np.arange(4, 15)
bit_error_rate = np.zeros(number_of_tests * len(snr))
bit_error_rate = np.reshape(bit_error_rate, (len(snr), number_of_tests))
packet_error_rate = np.zeros(number_of_tests * len(snr))
packet_error_rate = np.reshape(packet_error_rate, (len(snr), number_of_tests))

for err in range(len(snr)):
    for n_t in range(number_of_tests):
                
        # AWGN generation. Генерирование аддитивного белого гауссовского шума
        snr_dB = snr[err]  # Signal-to-noise ratio (SNR), deci-bells [dB]. Отношение сигнал-шум на поднесущую в децибелах (дБ)    
        noise_power = 10 ** (-0.1 * snr_dB)
        
        # Генерируем стандартный гауссовский шум, т.е., с единичной дисперсией и нулевым средним
        awgn = rng.standard_normal((n_symb + 1, n_fft + n_cp)) + 1j * rng.standard_normal((n_symb + 1, n_fft + n_cp))
        awgn *= np.sqrt(noise_power * 0.5)  # Нормализуем шум, чтобы получился шум с заданной мощностью шума
        
        # Добавляем шум к сигналу
        sig_noised = sig_cp_conv + awgn
        
        # Get rid of CP. Избавляемся от ЦП
        sig_wo_cp = sig_noised[:, n_cp : n_cp+n_fft :]
                
        # OFDM демодуляция через применение БПФ, чтобы вернуть сигнал обратно в частотную область.
        sig_fft = np.fft.fft(sig_wo_cp, n= n_fft, norm='ortho')
        
        # Удаление нулевых входов 
        sig_rx_ft = np.concatenate([sig_fft[:, 38:], sig_fft[:, 1:27]], axis=1)
            
        # Первый столбец выходного сигнала
        output_sig = sig_rx_ft[:1, :]
        
        # Оценка канала     
        channel_estimate = output_sig/fix_sig_qpsk
        sig_rx_ft /= channel_estimate
                
        # Удаление фиксированног столбца
        sig_without_fix = sig_rx_ft[1:, :]

        # Разворачиваем сигнал из двумерного в одномерный массив
        sig_fft = np.ravel(sig_without_fix)
           
        # Log Likelihood ratio (отношение правдоподобия)
        llr = np.zeros(len(sig_fft) * 2)        
        for i in range(len(sig_fft)):
            llr[i * 2] = (abs(sig_fft[i].real + np.sqrt(0.5))**2 - abs(sig_fft[i].real - np.sqrt(0.5))**2) 
            llr[i * 2 + 1] = (abs(sig_fft[i].imag + np.sqrt(0.5))**2 - abs(sig_fft[i].imag - np.sqrt(0.5))**2) 
             
        # Инициализация декодера Витерби
        num_states = 2**(m - 1)  # Количество состояний в решетке
        next_state = np.zeros((num_states, 2), dtype=int)  # Таблица следующих состояний
        for i in range(num_states):
            next_state[i] = [(i >> 1), (i >> 1) | (1 << (m - 2))]
        output_table = np.zeros((num_states, 2, 2), dtype=int)  # Таблица выходов на основе входного бита и текущего состояния
        
        for s in range(num_states):
            # Преобразуем номер состояния в битовую последовательность
            state_bits = np.array([int(x) for x in np.binary_repr(s, width = m - 1)], dtype=np.uint8)
            for b in [0, 1]:
                # Включаем входной бит в state_bits и вычисляем паритетные биты
                total_state = np.r_[[b], state_bits]                
                output_table[s][b][0] = np.sum(total_state[[0, 2, 3, 5, 6]]) % 2  # Генераторный полином на самом деле 171 в восьмеричной системе (позиции 0, 1, 3, 5, 6)
                output_table[s][b][1] = np.sum(total_state[[0, 1, 2, 3, 6]]) % 2
                           
        for s in range(num_states):
            for b in [0, 1]:
                output_table[s][b][0] = 1 if output_table[s][b][0] == 0 else -1
                output_table[s][b][1] = 1 if output_table[s][b][1] == 0 else -1

        # Процесс декодирования Витерби
        path_metric = np.full((2, num_states), np.inf)  # Две строки для отслеживания предыдущих и текущих путевых метрик
        path_metric[0][0] = 0  # Начинаем с состояния 0, поэтому метрика равна 0
        paths = np.zeros((n_used * n_symb * 2, num_states), dtype=int)  # Пути для каждого состояния  
        previous_state = np.zeros((n_used * n_symb * 2, num_states), dtype=int)  # Для восстановления декодированного пути  
        
        # Процесс декодирования Витерби
        for t in range(n_used * n_symb):   
            current_bits = llr[2 * t : 2 * (t + 1)]
            path_metric[1, :] = np.inf  # Сброс второй строки для новой итерации
            for state in range(num_states):
                for bit in [0, 1]:
                    next_s = next_state[state][bit]
                    metric = path_metric[0, state] + branch_metric(current_bits, state, bit)                    
                    if metric < path_metric[1, next_s]:
                        path_metric[1, next_s] = metric
                        paths[t + 1, next_s] = bit
                        previous_state[t + 1, next_s] = state
            path_metric[0] = path_metric[1]  # Копирование рассчитанных путевых метрик
       
        # Восстановление пути для декодирования последовательности
        decoded_bits = np.zeros(n_used * n_symb, dtype=np.uint8)
        state = np.argmin(path_metric[0])  # Находим состояние с наименьшей метрикой
        for t in range(n_used * n_symb - 1, -1, -1):            
            decoded_bits[t] = paths[t + 1, state]
            state = previous_state[t + 1, state]
               
        # Количество ошибочно принятых бит
        count_bit_error = 0    
        for i in range(len(decoded_bits)):            
            if decoded_bits[i] != input_bits[i]:
                count_bit_error += 1    
        # BER        
        bit_error_rate[err, n_t] = count_bit_error/ (n_used * n_symb)
        
        # PER
        packet_error_rate[err, n_t] = 0 if count_bit_error == 0 else 1
        
        
ber_average = np.zeros(len(snr))
per_average = np.zeros(len(snr))
for i in range(len(snr)):
    ber_average[i] = bit_error_rate[i, :].sum()
    ber_average[i] /= number_of_tests    
    per_average[i] = packet_error_rate[i, :].sum()
    per_average[i] /= number_of_tests


print(ber_average)   
print("****************************************")
print(per_average)

# Рисуем картинки
fig1, ax_1 = plt.subplots()
ax_1.plot(snr, ber_average)
ax_1.set(xlabel = 'ОСШ дБ', ylabel = 'Pbit')
plt.xscale('linear')
plt.yscale('log')
ax_1.grid(True)

fig2, ax_2 = plt.subplots()
ax_2.plot(snr, per_average) 
ax_2.set(xlabel = 'ОСШ дБ', ylabel = 'Ppkt') 
plt.xscale('linear')
plt.yscale('log')
ax_2.grid(True)

plt.show()  