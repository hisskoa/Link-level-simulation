# Link-level-simulation
Software implementation of convolutional channel encoding and decoding using the Viterbi algorithm.

1. Tests with additive white Gaussian noise.  For a packet of 1000 bits, for each value of the SNR (signal-to-noise ratio) (4 dB, 12 dB) for QPSK and a convolutional encoder with a structure as in
IEEE802.11a, and with the Viterbi decoder, 500 tests were conducted. Bit and packet errors are calculated for each test.

  ![Испытания с аддитивным белым гауссовским шумом(BER, CodeRate 0,5)](https://github.com/hisskoa/Link-level-simulation/assets/96256575/4c7db6e2-21f5-4b78-804c-4bc123a89dd3)

  ![Испытания с аддитивным белым гауссовским шумом(PER, CodeRate 0,5)](https://github.com/hisskoa/Link-level-simulation/assets/96256575/ebde84bf-db6a-4b36-bc6f-f0f9b9e599c0)

2. The results of comparing the hard and soft Viterbi decoder. For a packet of 2080 bits, for each value of the SNR (signal-to-noise ratio) (4 dB, 18 dB) for QPSK and a convolutional encoder with a structure as in
IEEE802.11a, and with the Viterbi decoder, 500 tests were conducted. Bit and packet errors are calculated for each test.

  ![Результаты сравнения мягкого и жесткого декодера Витерби(BER)](https://github.com/hisskoa/Link-level-simulation/assets/96256575/b55bd889-df40-4d00-967e-f790785dbba6)

  ![Результаты сравнения мягкого и жесткого декодера Витерби(PER)](https://github.com/hisskoa/Link-level-simulation/assets/96256575/5df39890-2dbc-457a-b3db-3c4a4b84be40)

3. Tests with different encoding speeds. For a packet of 1000 bits, for each value of the SNR (signal-to-noise ratio) (6 dB, 15 dB) for QPSK, a convolutional encoder with a structure like in IEEE 802.11a, for a channel with additive white Gaussian noise (AWGN) and with a Viterbi decoder, 100 tests were conducted. Bit and packet errors are calculated for each test.

  ![Испытания с различными скоростями кодирования(BER)](https://github.com/hisskoa/Link-level-simulation/assets/96256575/58b2a60d-f6a3-422d-a3bc-fc7fc5e63030)
<p align="center">
  ![Испытания с различными скоростями кодирования(PER)](https://github.com/hisskoa/Link-level-simulation/assets/96256575/eb953bf3-d5c9-4880-848d-7e89c28f3289)
</p>
 






