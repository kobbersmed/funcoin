# Release Instructions

1. Go into the GitHub repo directory on your local computer.

```
cd funcoin_release
```

2. Get the latest code on the main branch.

```
git checkout main
git pull
```

3. Update the version number in `setup.cfg` and save.

4. Commit the new version number and push to the remote repo.

```
git add setup.cfg
git commit -m "New version vX.Y.Z"
git push
```

5. Go to the GitHub website and create a release. Create a tag called vX.Y.Z and call the release vX.Y.Z. Add a description of the changes made to the repo since the last release.

6. Go back to the `funcoin_release` directory on your local computer.

7. If there's a directory called `dist`, delete it.

```
rm -r dist
```

8. Build the package.

```
python -m build
```

9. Upload to PyPI.

```
twine uplaod dist/*
```
