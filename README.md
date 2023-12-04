# FFR120 - Voting simulator project

This repo contains the code for simulating voting systems as part of a class project for FFR120.

### Experiments

This matrix corresponds to the experiments were are planning to run and wether or not the simulator currently supports doing so. Completed means that results of simulation are captured in a version controlled notebook.

| voting system                      | electorate                  | dispersion     | supported                | completed                |
| ---------------------------------- | --------------------------- | -------------- | ------------------------ | ------------------------ |
| plurality                          | centered, bipolar, tripolar | None, high     | <ul><li>- [x] </li></ul> | <ul><li>- [ ] </li></ul> |
| majority                           | centered, bipolar, tripolar | None, high     | <ul><li>- [x] </li></ul> | <ul><li>- [ ] </li></ul> |
| ranked choice                      | centered, bipolar, tripolar | None, high     | <ul><li>- [x] </li></ul> | <ul><li>- [ ] </li></ul> |
| plurality, majority, ranked choice | swedish public              | not applicable | <ul><li>- [ ] </li></ul> | <ul><li>- [ ] </li></ul> |

### Getting started

```
# install dependencies
python3 -m pip install -r requirements.txt

# run the simulation for a given voting system, and visualise
python3 src/vsim/cli.py \
    --voting-system plurality \
    --scenario polarized \
    --candidates 2 \
    --population 10_000 \
    --debug \
    --log debug

# run with more settings, logging and graphical output
python3 src/vsim/cli.py \
    --voting-system majority \
    --candidates 5
    --population 10_000
```

Supported systems of voting are currently:

- plurality
- majority

### Contributing

To contribute, make a branch, commit and push your changes and then create a pull request to merge to the develop branch. Be sure to integrate the latest develop version into your branch.

1. Go to github and clone the code
2. Create a new branch

```bash
git checkout -B my-feature-branch
```

3. Add your changes

```
git add .
git commit -am "description of my changes"
git push -u origin my-feature-branch
```

4. Navigate to (GitHub)[https://github.com/Expan75/complex-simulation-project/pulls] and create a new pull request under the branch to merge into main.

5. Complete code review and your done!
