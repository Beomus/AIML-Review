# Imagenette from [FastAI](https://github.com/fastai/imagenette)

Data can be downloaded manually [here](https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz) and extract into `data` folder.

Instructions:

```bash
mkdir data
cd data
curl -s https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz -o imagenette2-320.tgz
tar -xvzf imagenette2-320.tgz
```

Labels for imagenette

```json
{"n01440764": "tench", "n02102040": "english_springer", "n02979186": "cassette_player", "n03000684": "chain_saw", "n03028079": "church", "n03394916": "french_horn", "n03417042": "garbage_truck", "n03425413": "gas_pump", "n03445777": "golf_ball", "n03888257": "parachute"}
```

Please create a file called `data/imagenette_labels.json`.
