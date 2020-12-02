# Anaconda Cloud package uploader
# 
# Runs but doesn't attempt the upload unless 
# 
#  * the package version in meta.yaml is {% version = "N.N.N" %} 
# 
#  * the current build is a tag build vN.N.N such as triggered by a
#    github release

# some guarding
if [[ -z ${CONDA_DEFAULT_ENV} ]]; then
    echo "activate a conda env before running conda_upload.sh"
    exit -1
fi

# meant for a TravisCI deploy environment but easily tricked into running locally
# by setting these
if [[ "$TRAVIS" != "true" || -z "$TRAVIS_BRANCH" || -z "${PACKAGE_NAME}" ]]; then
    echo "conda_upload.sh is meant to run on TravisCI"
    exit -2
fi

# as in .travis.yml or use bld_prefix=${CONDA_PREFIX} for local testing
# bld_prefix="$HOME/miniconda3"
bld_prefix="/home/travis/miniconda"

# on TravisCI there should be a single linux-64 package tarball. insist
tarball=`/bin/ls -1 ${bld_prefix}/conda-bld/linux-64/${PACKAGE_NAME}-*-*.tar.bz2`
n_tarballs=`echo "${tarball}" | wc -w`
if (( $n_tarballs != 1 )); then
    echo "found $n_tarballs $PACKAGE_NAME tarballs there must be exactly 1"
    echo "$tarball"
    exit -3
fi

# whatever version string was set in the conda meta.yaml
version=`echo $tarball | sed -n "s/.*${PACKAGE_NAME}-\(.\+\)-.*/\1/p"`

# extract the numeric major.minor.patch portion of version, possibly empty
mmp=`echo $version | sed -n "s/\(\([0-9]\+\.\)\{1,2\}[0-9]\+\).*/\1/p"`

# uploads to conda cloud work like this:
#
# * github commits on branch dev w/ version M.N.P.devX, M.N.P.devX+1, ... upload
#   to conda label pre-release. For testing do this:
# 
#        conda install fitgrid -c kutaslab/label/pre-release
#
# * github commits on branch dev w/ version M.N.P *also* upload to pre-release, this
#   is the final CI check before PR dev -> main for a stable release.
#
# * manual github release tagged vM.N.P on main branch upload to
#   conda label main (and PyPI). This makes version M.N.P the default for
#
#         conda install fitgrid -c kutaslab
#

label="dry_run"
if [[ $TRAVIS_BRANCH = "dev" && "${version}" =~ ^${mmp}(.dev[0-9]+){0,1}$ ]]; then
    label="pre-release"
elif [[ "${version}" = "$mmp" && $TRAVIS_BRANCH = v$mmp ]]; then
    label="main"
fi

# build for multiple platforms ... who knows it might work
mkdir -p ${bld_prefix}/conda-convert/linux-64
cp ${tarball} ${bld_prefix}/conda-convert/linux-64
cd ${bld_prefix}/conda-convert
conda convert -p linux-64 -p osx-64 -p win-64 linux-64/${PACKAGE_NAME}*tar.bz2

# POSIX trick sets $ANACONDA_TOKEN if unset or empty string 
ANACONDA_TOKEN=${ANACONDA_TOKEN:-[not_set]}
conda_cmd="anaconda --token $ANACONDA_TOKEN upload ./**/${PACKAGE_NAME}*.tar.bz2 --label ${label}"

# thus far ...
echo "conda meta.yaml version: $version"
echo "package name: $PACKAGE_NAME"
echo "conda-bld: ${bld_prefix}/conda-bld/linux-64"
echo "tarball: $tarball"
echo "travis tag: $TRAVIS_TAG"
echo "travis branch: $TRAVIS_BRANCH"
echo "conda_label: ${label}"
echo "conda upload command: ${conda_cmd}"
echo "platforms:"
echo "$(ls ./**/${PACKAGE_NAME}*.tar.bz2)"

# if the token is in the ENV and 
#    attempt the upload 
# else
#    skip the upload and exit happy
# 
# conda upload knows the destination from the token

#if [[ $ANACONDA_TOKEN != "[not_set]"  ]]; then
if [[ $ANACONDA_TOKEN != "[not_set]" && ( $label = "main" || $label = "pre-release" ) ]]; then

    echo "uploading to Anconda Cloud: $PACKAGE_NAME$ $version $TRAVIS_BRANCH $label ..."
    conda install anaconda-client
     if ${conda_cmd}; then
    	echo "OK"
    else
    	echo "Failed"
    	exit -5
    fi
else
    echo "$PACKAGE_NAME $TRAVIS_BRANCH $version conda_upload.sh dry run ... OK"
fi
exit 0

