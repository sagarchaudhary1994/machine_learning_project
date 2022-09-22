# machine_learning_project

End to end machine learning project.

Requirement for this project-

1. Github Account
2. Heroku Account
3. VS Code IDE
4. Git CLI

Creating conda env

```
conda create -p venv python==3.7 -y
```

Activate conda env

```
conda activate venv/
```

Install requirements.txt

```
pip install -r requirements.txt
```

```
pip install -e .
```

Add files to git

```
git add .
```

Add a file to git

```
git add <filename>
```

Check the git status

```
git status
```

Check the git log

```
git log
```

Commit changes to local repo

```
git commit -m <commit message>
```

Push your changes to remote repo

```
git push origin <current branch>
```

Check remote url

```
git remote -v
```

Check the current branch

```
git branch
```

Set Up CI/CD on Heroku app, we need 3 information to set this up

1. HEROKU_EMAIL = <>
2. HEROKU_API_KEY = <>
3. HEROKU_APP_NAME = <>

Build Docker Image

```
docker build -t <image-name>:<tag-name> .
```

> Note: image name for the docker should be the lower case

To list docker images

```
docker images
```

Run docker image

```
docker run -p 5000:5000 -e PORT=5000 <Image Id>
docker run -p 5000:5000 -e PORT=5000 b3abf9a2e2ab
```

To check running containers in docker

```
docker ps
```

TO stop the docker container

```
docker stop <container_id>
```

Install module to run jupyter notebook in VS Code

```
install ipykernel
```

Install module to run YAML file in jupyter notebook

```
pip install PyYAML
```

Data Drift:
When your dataset stats got changed over time, we call it as data drift.
