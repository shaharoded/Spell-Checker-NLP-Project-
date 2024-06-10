# Noisy Channel Spell Check #

The code implements a context sensitive spell checker based on the Noisy Channel 
framework and the Norvig spell checker.

In order to determine the best correction, the algorithm will integrate the common error matrix, combined with a trained language model.

You'll load your (big) text file, and an error table (both examples are attached in the folder). Be sure to understand how the error table is made, if you wish to add to it.

Update tests.py file to initiate a model based on your picked files (modify the file_path and imports). You can run the file as is and test the different functionalitis in the flow.
