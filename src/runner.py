


#load nn config from file, pretrained

#start_page = TODO
#end_page = TODO

def run(start_page, end_page, config):
    page_history = []
    #start timer
    current_page = start_page
    page_history.append(current_page)
    pages_visited = 0
    while current_page != end_page:
        closest_hyperlink = find_closest_hyperlink(current_page, end_page, config) #TODO use some algorothm to find the closest hyperlink within our vocabulary
        print(f"Visiting page: {closest_hyperlink} \n")
        page_history.append(closest_hyperlink)
        pages_visited += 1
    #end timer
    #return page_history, timer

def find_closest_hyperlink(current_page, end_page, config):
    pass

if __name__ == "__main__":
    start_page = "Python (programming language)"
    end_page = "Artificial intelligence"
    config = {} #load config from file
    run(start_page, end_page, config)
    