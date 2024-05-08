# Link-level-simulation
Программная реализация сверточного канального кодирования и декодирования по алгоритму Витерби.

1. Для пакета из 1000 бит, для каждого значения ОСШ (отношение сигналшум) (4 dB, 12 dB) для QPSK и сверточного кодера со структурой, как в IEEE802.11a, и с декодером Витерби, проводилось 500 испытаний. Для каждого испытания посчитаны битовая и пакетная ошибки.

<p align="center">
  <img width="400" height="250" src="Tests with additive white Gaussian noise(BER, CodeRate 0,5).png">
  <img width="400" height="250" src="Tests with additive white Gaussian noise(PER, CodeRate 0,5).png">
</p>

2. Для пакета из 1000 бит, для каждого значения ОСШ (отношение сигналшум) (6 dB, 18 dB) для QPSK и сверточного кодера со структурой, как в IEEE802.11a, и с декодером Витерби, проводилось 1000 испытаний. Для каждого испытания посчитаны битовая и пакетная ошибки.

<p align="center">
  <img width="400" height="250" src="Results of comparing soft and hard Viterbi decoder(BER).png">
  <img width="400" height="250" src="Results of comparing soft and hard Viterbi decoder(PER).png">
</p>
  
3. Для пакета из 1000 бит, для каждого значения ОСШ (отношение сигнал-шум) (6 dB, 15 dB) для QPSK, сверточного кодера со структурой, как в IEEE802.11a, для канала с аддитивным белым гауссовским шумом (AWGN) и с декодером Витерби, проводилось 100 испытаний. Для каждого испытания посчитаны битовая и пакетная ошибки.

<p align="center">
  <img width="400" height="250" src="Tests with different coding rates(BER).png">
  <img width="400" height="250" src="Tests with different coding rates(PER).png">
</p>
 





