
# Opinions

https://neptune.ai/blog/how-to-organize-deep-learning-projects-best-practices

data/
    raw/ # The original immutable data dump
    interem/ # Intermediate data that has been transformed
    cropped/ # cropped and aligned
    processed/ # The final, connonical data sets for modeling 


data/
    raw/
        seals_2020/
            seal_1.jpg
        seals_2021/
        seals_2022/
    processed/
        images/ -> SDKJSA21903i - "Combined 2020-2022 data and cropped"
        images/ -> DSJASA21903i - "Combined 2022 data and cropped differently"

oxen clone hub.oxen.ai/Orlando/BasketballData:my-branch/processed/images

- Be aware of what is missing
    - Maybe blank directory, or metadata
    - Maybe look at what dropbox does for files in the cloud

oxen status
    - Show and mark whats missing

    raw/ (missing)
    processed/
        images/