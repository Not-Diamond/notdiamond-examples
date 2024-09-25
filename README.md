# notdiamond-examples

## Getting started

```shell
pyenv virtualenv 3.11 notdiamond-examples
pyenv activate notdiamond-examples
```

Install the dependencies:

```shell
poetry install
```

## Exploring Not Diamond

Create a `.env` file from the included template, making sure to populate keys for the providers
you would like to use. Then run the app:

```shell
streamlit run notdiamond_examples/streamlit/main.py
```

The app suggests some models for you to use, but you can also edit the app code to add [any
model supported by Not Diamond][supported].

## Chat with Not Diamond

Want to test out image generation? Watch the most popular models battle in Arena Mode? [Chat with Not Diamond].

<p align="center">
  <img src="./chat_nd.png" alt="Arena Mode in Chat">
</p>

## Support

For support please check out the [docs] or send us a [message].

[supported]: https://notdiamond.readme.io/docs/llm-models
[docs]: https://notdiamond.readme.io
[message]: mailto:support@notdiamond.ai
[Chat with Not Diamond]: https://chat.notdiamond.ai