# GPT2 sentiment classification finetune

Somewhat embarrassingly, before today, I hadn't ever finetuned an LLM. To correct this, I [vibe finetuned](https://x.com/karpathy/status/1886192184808149383) GPT-2 Small, a tiny (124M) model, to classify text as positive or negative. I used the [SST-2 dataset](https://huggingface.co/datasets/stanfordnlp/sst2). I added a binary classification head to GPT-2, and froze all but the last two of its 11 layers. I fed the classification head the last token's logits.

The modified model got a validation accuracy of 89.79% after three epochs, with the entire SST-2 training dataset seen each epoch (shuffled, of course). I did inference with some real Letterboxd reviews:

```bash
Sample text: 'As personal and egoless as you could ever hope to expect from an $120 million self-portrait that doubles as a fable about the fall of Ancient Rome, Francis Ford Coppola’s “Megalopolis” is the story of an ingenious eccentric who dares to stake his fortune on a more optimistic vision for the future — not because he thinks he can single-handedly bring that vision to bear, but rather because history has taught him that questioning a civilization’s present condition is the only reliable hope for preventing its ruin. Needless to say, the movie isn’t arriving a minute too soon.'
Predicted sentiment: negative
Sample text: 'YALL ARE WRONG You see you gotta go into the movie the same way as Francis Ford Coppola, blazed out of your fucking mind. You gotta meet him on the same plane, the same level of thinking. Hitting the cart 6 times before entering the showing like I did.'
Predicted sentiment: negative
Sample text: 'the 138-minute cinematic equivalent of someone showing you a youtube video they promise is really good'
Predicted sentiment: positive
Sample text: 'Dropping the kids off at Oppenheimer so the adults could watch Barbie'
Predicted sentiment: negative
Sample text: 'Hi Barbie.'
Predicted sentiment: positive
Sample text: 'As a work of and about plasticity, Barbie succeeds with vibrant, glowing colors. This is a triumph of manufactured design, with its expansive dollhouses and fashion accessories complimenting the mannered, almost manicured narrative journey of toys and humans.'
Predicted sentiment: positive
Sample text: 'I’m sorry to talk about a man when this is very much a movie about women but every second Gosling is onscreen is so funny. Even when he’s just standing there not talking it’s funny.'
Predicted sentiment: positive
Sample text: 'A ridiculous achievement in filmmaking. An absurdly immersive and heart-pounding experience. Cillian Murphy is a fucking stud and RDJ will be a front-runner for Best Supporting Actor. Ludwig Göransson put his entire nutsack into that score, coupled with a sound design that made me feel like I took a bomb to the chest.'
Predicted sentiment: positive
```

Some analysis:

For the reviews which obviously swing in one direction or the other, GPT-2 does
an okay job classifying them. For example, it classifies the last review
accurately. But for less obvious reviews (e.g., the second one, which is clearly
a positive review of Megalopolis), GPT-2 misclassifies. Is this because the
classification model is undertrained, or because GPT-2 itself is not smart
enough? To test this, I pasted in the review in the GPT-2 inference server at
https://huggingface.co/openai-community/gpt2, added in the words "This movie
was", then looked at GPT-2's completion. (I could not just ask GPT-2 if the
review was positive or negative because it's not an instruction-tuned model.) It
said:

```
YALL ARE WRONG You see you gotta go into the movie the same way as Francis Ford Coppola, blazed out of your fucking mind. You gotta meet him on the same plane, the same level of thinking. Hitting the cart 6 times before entering the showing like I did. This movie was a total fucking disaster. I was so fucking sick of it. I was so sick of it.
```

GPT-2 calls the film a "total fucking disaster", so the model really isn't smart
enough to know that review was positive! I asked Claude 3.7 Sonnet, a much
smarter model, if the review was positive or negative (I literally asked it if
was positive or negative), and it said it "appears to be positive, despite its
unconventional style". The bigger model gets it.
