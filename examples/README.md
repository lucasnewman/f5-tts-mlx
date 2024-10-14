
## Usage

To run the script, use the following format:

```bash
python generate.py --text "Your input text here" --duration 10
```

## Required Parameters

`--text`

string

Provide the text that you want to generate.

`-–duration`

float

Specify the length of the generated audio in seconds.

## Optional Parameters

`--model`

string, default: "lucasnewman/f5-tts-mlx"

Specify a custom model to use for generation. If not provided, the script will use the default model.

`--ref-audio`

string, default: "tests/test_en_1_ref_short.wav"

Provide a reference audio file path to help guide the generation.

`–-ref-text`

string, default: "Some call me nature, others call me mother nature." 

Provide a caption for the reference audio.


`-–output`

string, default: "output.wav"

Specify the output path where the generated audio will be saved. If not specified, the script will save the output to a default location.


`-–sway-coef`

float, default: 0.0

Set the sway sampling coefficient. The best values according to the paper are in the range of [-0.4...0.4].


`-–seed`

int, default: None (random)

Set a random seed for reproducible results.
