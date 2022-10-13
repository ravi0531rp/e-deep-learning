## Docs

* [Model Path](https://drive.google.com/file/d/1zIfDnuFygnWvo1zPOoTKYStNZQaFjExy/view?usp=sharing)

```
import gdown
gdown.download("https://drive.google.com/file/d/1zIfDnuFygnWvo1zPOoTKYStNZQaFjExy/view?usp=sharing", fuzzy = True)
```

## Commands

```
docker build -t app/fastapi .
```

```
docker run -it -p 8001:5001 -e MODEL_TYPE=horses_vs_humans -e NUM_WORKERS=4 app/fastapi
```
