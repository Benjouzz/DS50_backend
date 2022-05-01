from filtering import ContentBasedFiltering

def getReco():
    filtering = ContentBasedFiltering()
    filtering.setFilterBase(filter_base=27421523)
    filtering.loadData()
    filtering.processData()
    filtering.filter()

    return