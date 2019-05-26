# Data preprocess
## Crawl Safebooru
eg:
`http://safebooru.org/index.php?page=post&s=view&id=2844551`

Crawl Image from id=0 to current max id and collect tag meta information which appears in the html code, such as `class="tag-type-copyright"`
, `tag-type-general` or `tag-type-character`, write them in a `ori_tags.csv` whose table head includes:
`id,img_src,tags,types`

## Find most popular 512 attribute tags
1. Scan over the whole original csv and encode each tag into a tag index, forming functions `tag2index` `index2tag`
2. Count all tags using `Counter`, as our task only focus on background removal, just find the most popular 512 `general`(which I call it `attribute`) tags
3. Filter out those images which don't include these 512 tags
4. Reindex these tags to 0~511
5. Construct a dict which maps `image_id` to `attr_index`. And we should be aware of those pictures which are not successful downloaded, they cound cause pytorch dataloader exception, so we need to clean them by using `os.path.exists`.Finally, we cache it to `img_id2attr.pkl`, so that the training set is produced

# Dataset
Read the mentioned `img_id2attr.pkl`,cast the index into one-hot encoding and start training.