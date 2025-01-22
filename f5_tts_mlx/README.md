## Required Parameters

`--text`

string

Provide the text that you want to generate.

## Optional Parameters

`--duration`

float

Specify the length of the generated audio in seconds.

`--estimate-duration`

bool, default: false

If true, estimate the duration using a heuristic based on the text instead of the duration predictor model.

`--speed`

float, default: 1.0

Speaking speed modifier, used when an exact duration is not specified.

`--model`

string, default: "lucasnewman/f5-tts-mlx"

Specify a custom model to use for generation. If not provided, the script will use the default model.

`--ref-audio`

string, default: "tests/test_en_1_ref_short.wav"

Provide a reference audio file path to help guide the generation.

`--ref-text`

string, default: "Some call me nature, others call me mother nature." 

Provide a caption for the reference audio.

`--output`

string, default: None

Specify the output path where the generated audio will be saved. If not specified, audio will play as it's generated.

`--cfg`

float, default: 2.0

Specifies the strength used for classifier free guidance

`--method`

str, default: "rk4"

Specify the sampling method for the ODE. Options are "euler", "midpoint", and "rk4".

`--steps`

int, default: 8

Specify the number of steps used to sample the neural ODE. Lower steps trade off quality for latency.

`--sway-coef`

float, default: -1.0

Set the sway sampling coefficient. The best values according to the paper are in the range of [-1.0...1.0].

`--seed`

int, default: None (random)

Set a random seed for reproducible results.

`--q`

int, default: None

Number of bits to use for quantization. 4 and 8 are supported.
