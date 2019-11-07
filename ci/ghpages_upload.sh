# ------------------------------------------------------------
# approach based on TravisCI gh_pages provider
# ------------------------------------------------------------

# ------------------------------------------------------------
# rebuild sphinx docs ... moved here from before_deploy:
# ------------------------------------------------------------
pip install sphinx sphinx_rtd_theme jupyter nbsphinx nbconvert!=5.4
conda install -c conda-forge pandoc
conda list
make -C docs html
touch docs/build/html/.nojekyll

# init with defaults in case ...
GITHUB_TOKEN=${GITHUB_TOKEN:-not_set}
TRAVIS_BRANCH=${TRAVIS_BRANCH:-null_branch}

# docs destination 
git_repo=github.com/${TRAVIS_REPO_SLUG}
doc_branch="gh-pages"

user=${PACKAGE_NAME}_ghpages_bot  
user_email="${user}@the.cloud.org"

# copy fresh docs to working dir
tmp_docs=/tmp/work/${TRAVIS_BRANCH}-$(date +%F-%N)
mkdir -p ${tmp_docs}
cp -rT docs/build/html $tmp_docs   # T option else dotfiles .nojeckyll left behind
# ls -lR $tmp_docs

# bop over to the working dir to make a bare repo and orphan branch
cd ${tmp_docs}
echo "working in $(pwd)"

git init
git config --local user.name $user
git config --local user.email $user_email
git config --local -l

echo "checking out orphan branch ${doc_branch} ..."
if git checkout --orphan ${doc_branch}; then
    echo "OK"
else
    echo "failed"
fi

# commit and push the new docs
git add -A
git commit -a -m 'TravisCI rebuilt docs for gh-pages'

# --------------------------------------------------------------
# push the docs to github ... intercept github replies which may
# expose the token in the URL to the travis log
# --------------------------------------------------------------
reply=`git remote add origin https://${user}:${GITHUB_TOKEN}@${git_repo} 2>&1`
echo ${reply//${GITHUB_TOKEN}/}  # strip github token(s) from fails

reply=`git push -u origin ${doc_branch} --force 2>&1`
echo ${reply//${GITHUB_TOKEN}/} 

echo "Done"
exit 0
