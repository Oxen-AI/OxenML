We've heard a couple times, and I have been thinking in my head..."People don't want to train models". Fine. I disagree that every task will be solved with one model to rule them all. But fine. Let's take that as a given. Because we have seen it over and over again, customers with the end product, do not want to train the model. 

What I do think is that your model will fail. That's just how the world works. I think that will never change. New information comes in. The model is not continually learning. It will fail. 

How do you capture these failures? How do you quantify them? And then, how do you set yourself up so boom, we can now fine tune our model. I think that is the initial value prop.

0) Big companies with lots of compute train a model, fine, take that as a given, it is expensive and takes a lot of time
1) Customers want to easily "try" a model (Hugging Face + Gradio)
2) Customers want to know will this work in my product (Some sort of Twilio style API integration....could be Oxen?)
    - Questions here about performance
        - Throughput, Latency, Accuracy, Precision, Recall
    - Is there a "one button buy bitcoin" type product here...that leads to 3?
    - One line, integrate model into product, pay as you go...hugging face is kind of setup to knock this out
3) Customer want to easily fix error cases, which starts with collecting all data (one line integration Oxen)
    - ox.save(...)
    - There was a moment where with speech recognition for example the guy was like "well shit that seems like a lot of effor to fix...and once we fix it for this one transcription, will it really apply again?"
        - If you are fixing it anyways, data/feedback should go back to the model owner
4) Model owner is shit out of luck if they had not been collecting the error cases, and the model running is shit out of luck if they can't communicate errors to the creator
5) How do you even begin to know how to solve the problem - Moriarty Error Analysis Tool - but for product managers. Be able to use human knowledge, and ability to generalize, to improve the model. Can they even extract features and rules? Hmm maybe that's a silly rabbit hole. Maybe not?

Thought from OpenAI guy (Zach Kass) "Extent to Companys that help incorporate OpenAI and LLMs, and test them… there’s a $xB Co there."

# Product V1:

One line integration - save data to Oxen Repo on branch in directory.

```python
import oxen as ox

repo = "http://hub.oxen.ai/g/speech"
branch = "deployed-to-hf"
directory = "wild"
input_blob = np.imread("path/to/image.jpg")
outputs = model.predict()
outputs_json = to_json(outputs)
ox.save(repo, branch, directory, input_blob, outputs_json)
```

Random other thoughts:

Woah woah woah....If GPT-3 or Dall-e starts generating content on the internet, then feeding in it's own content, we get a crazy loop that needs to be fixed - enter Oxen

https://www.theatlantic.com/ideas/archive/2020/09/future-propaganda-will-be-computer-generated/616400/

More Yann LeCun on why LLMs are not the end all be all...

https://www.zdnet.com/article/metas-ai-guru-lecun-most-of-todays-ai-approaches-will-never-lead-to-true-intelligence/

