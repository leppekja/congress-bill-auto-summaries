# congress-bill-auto-summaries

2021 Advanced ML Project to use abstract summarization on a set of Congressional Bills

## Set Up Environment

A requirements.txt file is available with the requisite packages necessary to collect the data and run the model.

## Repository Overview

Each folder has its own README with file descriptions. Please refer to folders for information on data collection and preprocessing, as well as supplementary model functions that were later incorporated into the Jupyter Notebooks on the top level of the repository.

Note that files in the Archive directory are variations and/or historical artifacts from current iterations of the models.

## Baseline Models

We adapted [NLP From Scratch: Translation with a Sequence to Sequence Model with Attention](https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html) from PyTorch. Credit to Sean Robertson for their authorship. Cells written by us are noted within the notebook; minor changes are made otherwise.

We implement an extractive summary method that pulls the official-title section from the bill as a comparable baseline for our abstractive model. This may be found in extractive_summary.py, and uses the structure of the XML file to parse the text from the bill.

## Encoder Decoder Model

    import model as md
    import load_data as ld
    import torch.optim as optim
    import torch

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    vocab = ld.build_vocab(training_data)
    glove = md.build_glove(vocab)
    # 300, 256, glove_vecs
    encoder = md.Encoder(input_features_length, hidden_size,  glove_vectors, device).to(device)
    # hidden size 256
    decoder = md.Decoder(len(vocab), hidden_size, glove_vecs).to(device)
    seq = md.Seq2Seq(encoder, decoder, device)
    adams = optim.Adam(encoder.parameters(), lr=.0001)
    md.train_an_epoch(seq, dataloaders_dict['train_data'], adams)

## References

#### Data Preprocessing

- [Stanford NLP Toolkit](https://www.aclweb.org/anthology/P14-5010.pdf)

#### Models

- [Sequence to Sequence Learning with Neural Networks](https://colab.research.google.com/github/bentrevett/pytorch-seq2seq/blob/master/1%20-%20Sequence%20to%20Sequence%20Learning%20with%20Neural%20Networks.ipynb#scrollTo=osmt4oYsCVgO)
- [Get To The Point: Summarization with POinter-Generator Networks](https://www.aclweb.org/anthology/P17-1099.pdf)
- [Summary Level Training of Sentence Rewriting for Abstractive Summarization](https://www.aclweb.org/anthology/D19-5402.pdf)
- [Abstractive Summarization Methods](https://medium.com/sciforce/towards-automatic-summarization-part-2-abstractive-methods-c424386a65ea)
- [Long Short-Term Memory Based Recurrent Neural Network](https://arxiv.org/pdf/1402.1128.pdf)
- [Text Summarization Using Neural Networks](https://digital.library.txstate.edu/bitstream/handle/10877/3819/fulltext.pdf)
- [Neural Text Summarization](https://cs224d.stanford.edu/reports/urvashik.pdf)
- [Text Summarization using RNN](https://iq.opengenus.org/text-summarization-using-rnn/)
