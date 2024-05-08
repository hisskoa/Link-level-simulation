# Link-level-simulation
Software implementation of convolutional channel encoding and decoding using the Viterbi algorithm.

1. Tests with additive white Gaussian noise.  For a packet of 1000 bits, for each value of the SNR (signal-to-noise ratio) (4 dB, 12 dB) for QPSK and a convolutional encoder with a structure as in
IEEE802.11a, and with the Viterbi decoder, 500 tests were conducted. Bit and packet errors are calculated for each test.

<p align="center">
  <img width="400" height="250" src="Tests with additive white Gaussian noise(BER, CodeRate 0,5).png">
  <img width="400" height="250" src="Tests with additive white Gaussian noise(PER, CodeRate 0,5).png">
</p>

2. The results of comparing the hard and soft Viterbi decoder. For a packet of 2080 bits, for each value of the SNR (signal-to-noise ratio) (4 dB, 18 dB) for QPSK and a convolutional encoder with a structure as in
IEEE802.11a, and with the Viterbi decoder, 500 tests were conducted. Bit and packet errors are calculated for each test.

<p align="center">
  <img width="400" height="250" src="Results of comparing soft and hard Viterbi decoder(BER).png">
  <img width="400" height="250" src="Results of comparing soft and hard Viterbi decoder(PER).png">
</p>
  
3. Tests with different encoding speeds. For a packet of 1000 bits, for each value of the SNR (signal-to-noise ratio) (6 dB, 15 dB) for QPSK, a convolutional encoder with a structure like in IEEE 802.11a, for a channel with additive white Gaussian noise (AWGN) and with a Viterbi decoder, 100 tests were conducted. Bit and packet errors are calculated for each test.

<p align="center">
  <img width="400" height="250" src="Tests with different coding rates(BER).png">
  <img width="400" height="250" src="Tests with different coding rates(PER).png">
</p>
 





