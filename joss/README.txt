To preview typeset paper.pdf 

* install docker
* open a bash shell
* run this in the repo top-level directory (= parent of ./joss)

     docker run --rm --volume $PWD/joss:/data --user $(id -u):$(id -g) --env JOURNAL=joss openjournals/paperdraft
