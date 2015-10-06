import scipy.io
import sys
import os
import pycurl


index = {}
index['WNID'] = 1
index['IMAGENETID'] = 0
index['WORDS'] = 2
index['HEIGHT'] = 6


def load_synsets(meta_fn):
    meta = scipy.io.loadmat(meta_fn)
    synsets = meta['synsets'][0]
    return synsets


def parse_height(synsets):
    synsets_height = [int(synsets[i][index['HEIGHT']][0])
                      for i in range(len(synsets))]
    return synsets_height


def build_index(synsets):
    synset_count = len(synsets)
    print "Totoal number of synsest = %d" % synset_count
    imagenet_id_list = [synsets[i][index['IMAGENETID']][0][0] - 1
                        for i in range(synset_count)]
    word_list = [synsets[i][index['WORDS']][0]
                 for i in range(synset_count)]
    wnid_list = [synsets[i][index['WNID']][0]
                 for i in range(synset_count)]
    return imagenet_id_list, word_list, wnid_list


def search_wnid(wnid, wnid_list, imagenet_id_list):
    # remove the '-' in the beginning of wnid
    wnid = wnid.strip('-')
    if wnid in wnid_list:
        return imagenet_id_list[wnid_list.index(wnid)]
    else:
        return -1


def search_wnids(wnids, wnid_list, imagenet_id_list,
                 min_id=0, max_id=1000):
    """Search imagenet id using wordnet_id

    @Parameters:
    min_id: the minimal number of id
    max_id: maximal number of id

    only those ids within the range [min_id, max_id) will be returned.
    """
    wnids = [search_wnid(wnid, wnid_list, imagenet_id_list)
             for wnid in wnids]
    wnids = [item for item in wnids if item >= min_id and item < max_id]
    return wnids


def getlist(input_fn, wnid_list, imagenet_id_list):
    """Generate a list of all synsets for the inputlist

    Return {category_wnid: imagenet_id list}
    """
    lines = []
    with open(input_fn, 'r') as f:
        for line in f:
            lines.append(line.strip())
    # the first line is the category wnid itself
    wnids = lines[1:]
    return search_wnids(wnids, wnid_list, imagenet_id_list)


def read_synsets(synsets_fn):
    synsets = load_synsets(synsets_fn)
    synset_heights = parse_height(synsets)
    imagenet_id_list, word_list, wnid_list = build_index(synsets)
    return imagenet_id_list, word_list, wnid_list, synset_heights


def generate_hyponym_list(wnid_list, synset_heights, height=7):
    """Geneate a list of synsets to partition the imagenet labels

    @Parameter:
        height: the height of synset in wordnet to select
    """
    return [wnid_list[i] for i in range(len(wnid_list))
            if synset_heights[i] == height]


def main(argv):
    """Task: read synsets from file, and search for synsets of given height
    Then save all files to a given folder, including:
    list of hyponyms
    list of imagenet_id for given hyponyms

    argv[1]: filename of synsets
    argv[2]: target folder
    argv[3]: [height]
    """

    if len(argv) > 4 or len(argv) < 3:
        raise Exception("""Usage:
        {} SYNSETS_FN TARGET_DIR [HEIGHT]""".format(argv[0]))
    else:
        synsets_fn = argv[1]
        target_dir = argv[2]
        if len(argv) == 4:
            height = int(argv[3])
        else:
            height = 7

    if not os.path.exists(target_dir):
        os.mkdir(target_dir)

    # Read synsets
    imagenet_id_list, word_list, wnid_list, synset_height = read_synsets(
        synsets_fn)
    # generate candidate list
    candidate_list = generate_hyponym_list(wnid_list, synset_height, height)

    with open(os.path.join(target_dir, 'list.txt'), 'w') as list_fp:
        for wnid in candidate_list:
            print wnid
            # download the hyponym
            fn = os.path.join(target_dir, "{}.txt".format(wnid))
            print("Task: downloading list for WordNetID {wnid} into {fn}"
                  .format(wnid=wnid, fn=fn))
            url = "http://www.image-net.org/api/text/"\
                  "wordnet.structure.hyponym?"\
                  "wnid={}&full=1".format(wnid)
            url = url.replace('\n', '')
            print url
            with open(fn, 'w') as synset_fp:
                c = pycurl.Curl()
                c.setopt(c.URL, url)
                c.setopt(c.WRITEDATA, synset_fp)
                c.perform()
                c.close()
            rslt = getlist(fn, wnid_list, imagenet_id_list)
            word = word_list[wnid_list.index(wnid)]
            # write information to list_fp
            list_fp.write("{wid}\t{word}\t{count}\n".format(
                wid=wnid, word=word, count=len(rslt)))
            # write list to file
            imagenet_id_fn = os.path.join(target_dir, '{}_list.txt'
                                          .format(wnid))

            with open(imagenet_id_fn, 'w') as imagenet_id_fp:
                imagenet_id_fp.write('\n'.join(
                    [str(id) for id in rslt]))

if __name__ == "__main__":
    sys.exit(main(sys.argv))
