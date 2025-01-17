<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Web server inference

A web server is basically a system that waits for requests and serves them as they come in. This means you can use [`Pipeline`] as an inference engine on the web server, since you can use an iterator (similar to how you would [iterate over a dataset](./pipeline_tutorial#large-datasets)) to handle each incoming request.

Designing a web server with [`Pipeline`] is unique though because they're fundamentally different. Web servers are multiplexed (multithreaded, async, etc.) to handle multiple requests concurrently. [`Pipeline`] and its underlying model on the other hand are not designed for parallelism because they take a lot of memory. It's best to give a [`Pipeline`] all the available resources when they're running or for a compute intensive job.

This guide shows how to work around this difference by using the web server to handle the light load of receiving and sending requests, and having a single thread to handle the heavier load of running [`Pipeline`].

## Create a server

[Starlette](https://www.starlette.io/) is a lightweight framework for building web servers. You can use any other framework you'd like, but you may have to make some changes to the code below.

Before you begin, make sure Starlette and [uvicorn](http://www.uvicorn.org/) are installed.

```py
!pip install starlette uvicorn
```

Now you can create a simple web server in a `server.py` file. The key is to only load the model **once** to prevent unnecessary copies of it from consuming memory.

Create a pipeline to fill in the masked token, `[MASK]`.

```py
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route
from transformers import pipeline
import asyncio

async def homepage(request):
    payload = await request.body()
    string = payload.decode("utf-8")
    response_q = asyncio.Queue()
    await request.app.model_queue.put((string, response_q))
    output = await response_q.get()
    return JSONResponse(output)

async def server_loop(q):
    pipeline = pipeline(task="fill-mask",model="google-bert/bert-base-uncased")
    while True:
        (string, response_q) = await q.get()
        out = pipeline(string)
        await response_q.put(out)

app = Starlette(
    routes=[
        Route("/", homepage, methods=["POST"]),
    ],
)

@app.on_event("startup")
async def startup_event():
    q = asyncio.Queue()
    app.model_queue = q
    asyncio.create_task(server_loop(q))
```

Start the server with the following command.

```bash
uvicorn server:app
```

The server can be queried now with a POST request.

```bash
curl -X POST -d "Paris is the [MASK] of France." http://localhost:8000/
[{'score': 0.9969332218170166,
  'token': 3007,
  'token_str': 'capital',
  'sequence': 'paris is the capital of france.'},
 {'score': 0.0005914849461987615,
  'token': 2540,
  'token_str': 'heart',
  'sequence': 'paris is the heart of france.'},
 {'score': 0.00043787318281829357,
  'token': 2415,
  'token_str': 'center',
  'sequence': 'paris is the center of france.'},
 {'score': 0.0003378340043127537,
  'token': 2803,
  'token_str': 'centre',
  'sequence': 'paris is the centre of france.'},
 {'score': 0.00026995912776328623,
  'token': 2103,
  'token_str': 'city',
  'sequence': 'paris is the city of france.'}]
```

## Queuing requests

The server's queuing mechanism can be used for some interesting applications such as dynamic batching. With dynamic batching, you can accumulate several requests first before processing them with the [`Pipeline`].

The example below is written in pseudocode for readability rather than performance, in particular, you'll notice that:

1. There is no batch size limit.
2. The timeout is reset on every queue fetch, so you could end up waiting much longer than the `timeout` value before processing a request. This would also delay the first inference request by that amount of time. The web server always waits 1ms even if the queue is empty, which is inefficient, because you could be using that time to start inference. It could make sense though if batching is essential to your use case.

    It would be better to have a single 1ms deadline, instead of resetting it on every fetch.

```py
(string, rq) = await q.get()
strings = []
queues = []
while True:
    try:
        (string, rq) = await asyncio.wait_for(q.get(), timeout=0.001)
    except asyncio.exceptions.TimeoutError:
        break
    strings.append(string)
    queues.append(rq)
strings
outs = pipeline(strings, batch_size=len(strings))
for rq, out in zip(queues, outs):
    await rq.put(out)
```

## Error checking

There are many things that can go wrong in production. You could run out-of-memory, out of space, fail to load a model, have an incorrect model configuration, have an incorrect query, and so much more!

Adding `try...except` statements could be helpful to return these errors to the user for debugging. Keep in mind that this could pose a security risk though if you shouldn't be revealing certain information.

## Circuit breaking

It is better to return errors when the server is overloaded instead of forcing a user to wait indefinitely. Try to return a 503 or 504 error instead of making a user wait for a really long time.

It is relatively simple to implement these error types since it's only a single queue. You should look at the queue size to determine when to start returning errors before your server fails under load.

## Block the main thread

PyTorch is not async aware, so computation will block the main thread from running.

For this reason, it's better to run PyTorch on its own separate thread or process. When inference of a single request is especially long (> 1s), it's even more important because it means every query during inference must wait 1s before even receiving an error.

## Dynamic batching

Dynamic batching can be very effective when used in the correct setting, but it's not necessary when you're only passing 1 request at a time (see [batch inference](./pipeline_tutorial#batch-inference) for more details).
