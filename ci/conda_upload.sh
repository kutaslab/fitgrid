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
if [[ "$TRAVIS" != "true" || -z "$TRAVIS_BRANCH" || -z "${PACKAGE_NAME}" ]]; then
    echo "conda_upload.sh is meant to run on TravisCI"
    exit -2
fi

# set the parent of conda-bld, the else isn't needed for TravisCI but 
# simplifies local testing
if [ $USER = "travis" ]; then
    bld_prefix="/home/travis/miniconda"  # from the .travis.yml
else
    bld_prefix=${CONDA_PREFIX}
fi

# on TravisCI there should be a single linux-64 package tarball. insist
tarball=`/bin/ls -1 ${bld_prefix}/conda-bld/linux-64/${PACKAGE_NAME}-*-*.tar.bz2`
n_tarballs=`echo "${tarball}" | wc -w`
if (( $n_tarballs != 1 )); then
    echo "found $n_tarballs package tarballs there must be exactly 1"
    echo "$tarball"
    exit -3
fi

# whatever version string was set in the conda meta.yaml
version=`echo $tarball | sed -n "s/.*${PACKAGE_NAME}-\(.\+\)-.*/\1/p"`

# extract the numeric major.minor.patch portion of version, possibly empty
mmp=`echo $version | sed -n "s/\(\([0-9]\+\.\)\{1,2\}[0-9]\+\).*/\1/p"`

# Are we building a release version according to the convention that
# releases are tagged vMajor.Minor.Release?
#
# * is $version = $mmp then version a strict numeric
#   Major.Minor.Patch, not further decorated, e.g., with .dev this or
#   rc that?
# 
# * is the tag vMajor.Minor.Patch (TravisCI treats tagged commits as a branch)?
if [[ "${version}" = "$mmp" && $TRAVIS_BRANCH = v$mmp ]]; then
    is_release="true"
    conda install anaconda-client
else
    is_release="false"
fi

# POSIX trick sets $ANACONDA_TOKEN if unset or empty string 
ANACONDA_TOKEN=${ANACONDA_TOKEN:-[not_set]}
conda_cmd="anaconda --token $ANACONDA_TOKEN upload ${tarball}"

# thus far ...
echo "conda meta.yaml version: $version"
echo "package name: $PACKAGE_NAME"
echo "conda-bld: ${bld_prefix}/conda-bld/linux-64"
echo "tarball: $tarball"
echo "travis branch: $TRAVIS_BRANCH"
echo "is_release: $is_release"
echo "conda upload command: ${conda_cmd}"

# if the token is in the ENV and this is a vN.N.N tagged commit
#    attempt the upload 
# else
#    skip the upload and exit happy
# 
# conda upload knows the destination from the token
if [[ $ANACONDA_TOKEN != "[not_set]" && $is_release = "true" ]]; then

    echo "uploading to Anconda Cloud: $PACKAGE_NAME$ $version ..."
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

