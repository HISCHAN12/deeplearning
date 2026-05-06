Student(s) name:22000708 Heechan Jung
Title: TFT LOL W/L prediction & recommendation system. 
Summary (200 words): Explain briefly the topic you chose, and what will you do in your report.
  This report proposes a deep learning framework designed to cluster "Teamfight Tactics" (TFT) meta-decks, predict final match placements, and provide strategic action recommendations. By utilizing core concepts from the Deep Learning textbook, this project aims to apply complex neural network architectures to highly non-linear, real-world gaming data.

First, I will collect large-scale match logs using the Riot Developer API. To handle the high-dimensional nature of champion and augment configurations, I will employ Autoencoders (Chapter 14) to compress this data into low-dimensional embedding vectors, which will seamlessly group similar in-game compositions. Crucially, by statistically analyzing these clustered decks, the model will identify specific in-game decisions—such as optimal positioning evaluated via Convolutional Neural Networks (CNNs, Chapter 9) or key unit acquisitions—that historically lead to higher win rates.

Finally, these comprehensive embeddings and spatial vectors will be fed into Deep Feedforward Networks (Chapter 6) to not only predict precise final rankings but also recommend the most statistically advantageous next steps for the player. Throughout this pipeline, I will rigorously apply regularization (Chapter 7) and advanced optimization strategies (Chapter 8) to prevent overfitting, ultimately building a reliable, actionable predictive AI model.
