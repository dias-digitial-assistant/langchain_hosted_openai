# Langchain with hosted OpenAI API

### Intro
This repo shares the code for accessing the hosted OpenAI API.
Each team gets a token (not OpenAI token, but token to the server), which can then be used to access the OpenAI models.

### Supported models
- gpt-3.5-turbo
- text-embedding-ada-002

### Steps
- Each team will be given a token and the URL, port of the server.
- Port forward to port 'X' on the local machine.
- Clone this repo
- Create a '''.env''' file in the same folder and define the variables.
- Run the test files to check Text generation and the Embedding generation.
