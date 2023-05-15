"""
6.1010 Spring '23 Lab 3: Bacon Number
"""

#!/usr/bin/env python3

import pickle

# NO ADDITIONAL IMPORTS ALLOWED!


def transform_data(raw_data):
    """
    Given database of a list of tuples (which indicate pairs of
    actors who have acted together in a given movie)
    Returns dictionary mapping an actor to a list of tuples
    containing each actor they have acted with, in the movie they
    acted in (not including themselves)
    """
    # dictionary of: actor -> set (no repeats) of tuples (actor, movie with actor)
    connections = {}

    # raw_data => (actor1, actor2, movie)
    for entry in raw_data:
        connections.setdefault(entry[0], set()).add((entry[1], entry[2]))
        connections.setdefault(entry[1], set()).add((entry[0], entry[2]))
    return connections


def acted_together(transformed_data, actor_id_1, actor_id_2):
    """
    Expected input is output of transform_data (dictionary of
    mappings between an actor and a tuple (the actor they acted
    with, the movie they acted in)
    """
    if actor_id_1 == actor_id_2:
        return True
    else:
        for listing in transformed_data[actor_id_1]:
            if listing[0] == actor_id_2:
                return True
    return False


def id_to_name(names_db, actor_id):
    """
    Returns name of actor with specified id

    names_db should be a dictionary
    mapping names to id numbers
    """
    for name in names_db.keys():
        if names_db[name] == actor_id:
            return name


def name_to_id(names_db, name):
    """
    Returns id of actor with specified name

    names_db should be a dictionary
    mapping names to id numbers
    """
    return names_db[name]


def id_to_movie(movies_db, movie_id):
    """
    Returns titles of movie with specified id

    movies_db should be a dictionary
    mapping movie titles to id numbers
    """
    for title in movies_db.keys():
        if movies_db[title] == movie_id:
            return title


def movie_to_id(movies_db, title):
    """
    Returns id of movie with specified title

    nmovies_db should be a dictionary
    mapping movie titles to id numbers
    """
    return movies_db[title]


def actors_with_bacon_number(transformed_data, n):
    """
    Uses a breadth first search to find
    all of the unweighted paths with length n
    """
    BACON_ID = 4724
    degree_n_actors = {BACON_ID}  # keeps track of degree n actors
    old_actors = {BACON_ID}  # keeps track of actors from lower bacon number levels
    new_actors = set()  # keeps track of new actors in next frontier

    # moving one frontier at a time:
    # transformed_data has info about a given actor's neighbors
    for _ in range(n):
        if len(degree_n_actors) == 0:  # no more actors to iterate through
            break
        else:
            for actor in degree_n_actors:
                new_actors.update(
                    {
                        neighbor[0]
                        for neighbor in transformed_data[actor]
                        if neighbor[0] not in old_actors
                    }
                )
                # cumulative set; actors should not come up twice in
                # this breadth first search, because the first tme
                # they are examined is the lowest path they have to
                # bacon; does not have repeats -> is a set
                old_actors.update({neighbor[0] for neighbor in transformed_data[actor]})
            degree_n_actors = new_actors
            new_actors = set()
    return degree_n_actors


def bacon_path(transformed_data, actor_id, comp_path_func=None):
    """
    Calls actor_to_actor path with
    Kevin Bacon's id as the starting actor

    *transformed_data must be a dictionary
    mapping each actor to a set of tuples
    documenting each actor they worked with,
    and the movie in which they acted

    Returns list of actor ids forming shortest path
    """
    BACON_ID = 4724
    return actor_to_actor_path(
        transformed_data, BACON_ID, actor_id, _comp_path_func=comp_path_func
    )


# old method
# def actor_to_actor_path(transformed_data, actor_id_1, actor_id_2):
#     actor_queue = [actor_id_1] #FIFO
#     curr_actor = None
#     checked = set()

#     #keeps track of a node's parent
#     path_parent = {actor_id_1:None}

#     while actor_queue:
#         curr_actor = actor_queue.pop(0)

#         if curr_actor == actor_id_2: #found
#             break
#         for neighbor in transformed_data[curr_actor]:
#             if neighbor[0] not in checked:
#                 actor_queue.append(neighbor[0])

#                 # I thought I needed the below code to
#                 # keep track of the right parents, but
#                 # because this is a breadth first search
#                 # for an unweighted graph, the first time
#                 # you encounter a node will be the shortest
#                 # path to that node, and a node is never
#                 # checked twice due to the if-statement above
#                 # helpful for weighted graph

#                 #Uneeded code#
#                 # finds smallest parent-frontier pair in the
#                 # case of returning to a node that was already visited
#                 # parent = [path_parent.setdefault(neighbor[0], \
#                   (None, float('inf'))), (curr_actor, path_parent[curr_actor][1] + 1)]
#                 # parent.sort(key=lambda x:x[1])
#                 # path_parent[neighbor[0]] = parent[0]

#                 # sets each neighbor's parent to be the curr_actor
#                 path_parent[neighbor[0]] = curr_actor
#                 checked.add(neighbor[0])

#     if curr_actor != actor_id_2:
#         return None

#     actor_path = [curr_actor]
#     while curr_actor != actor_id_1:
#         curr_actor = path_parent[curr_actor]
#         actor_path.append(curr_actor)

#     return actor_path[::-1]


# new method, implemented with general actor_path
def actor_to_actor_path(transformed_data, actor_id_1, actor_id_2, _comp_path_func=None):
    """
    Calls actor_path with goal function being
    that the end actor id is a certain id

    *transformed_data must be a dictionary
    mapping each actor to a set of tuples
    documenting each actor they worked with,
    and the movie in which they acted

    Returns list of actor ids forming shortest path
    """
    return actor_path(
        transformed_data,
        actor_id_1,
        lambda x: (x == actor_id_2),
        comp_path_func=_comp_path_func,
    )


def movie_path(transformed_data, actor_id_1, actor_id_2):
    """
    Calls actor_to_actor_path with function
    that finds movie path element, meaning that
    a list of movies which connect actor1 with actor2
    is returned
    """

    def movie_path_element(last_two_nodes):
        # list of movies that the next two
        # actors in the path acted in together
        title = [
            pair[1]
            for pair in transformed_data[last_two_nodes[1]]
            if pair[0] == last_two_nodes[0]
        ]
        return title[0]

    return actor_to_actor_path(
        transformed_data, actor_id_1, actor_id_2, _comp_path_func=movie_path_element
    )


def get_neighbors(curr_actor, transformed_data, checked):
    """
    Takes in data connecting actors and movies
    Returns a set of the current actor's neighbors
    if they are not in the checked set.

    *transformed_data should be a dictionary
    mapping actors to actors + movies
    """
    return {
        neighbor[0]
        for neighbor in transformed_data[curr_actor]
        if neighbor[0] not in checked
    }


def actor_path(
    transformed_data,
    actor_id_1,
    goal_test_function,
    get_neighbors_func=get_neighbors,
    comp_path_func=None,
):
    """
    Finds the shortest path between two
    actors, given an intial actor and a
    True/False function that the goal
    actor will satisfy

    *transformed_data must be a dictionary
    mapping each actor to a set of tuples
    documenting each actor they worked with,
    and the movie in which they acted

    Returns list of actor ids
    """
    # FIFO queue for breadth first search
    actor_queue = [actor_id_1]

    # keeps track of already visited actors,
    # because the first time you see an actor
    # in a BF search, you know it is the
    # result of the shortest path
    checked = set()
    curr_actor = None

    # dictionary mapping actors to the parent node
    # in the shortest path from start (in BF search
    # the shortest path, is the first one found,
    # and so the associated parent is the first
    # parent linked to a given actor
    path_parent = {actor_id_1: None}

    # while there are still actors
    # left to be searched in the graph
    while actor_queue:
        curr_actor = actor_queue.pop(0)  # actor of focus
        if goal_test_function(curr_actor):  # test if we found actor
            break
        # if not found actor, adds new neighbors to FIFO queue
        for neighbor in get_neighbors_func(curr_actor, transformed_data, checked):
            actor_queue.append(neighbor)
            checked.add(neighbor)

            # sets each neighbor's parent to be
            # the curr_actor (which is the parent
            # associated with the shortest path)
            path_parent[neighbor] = curr_actor

    # went through the entire graph without finding the actor
    if not goal_test_function(curr_actor):
        return None

    # otherwise, found actor and returns reconstructed
    # path from source to that actor

    return trace_back(
        actor_id_1, curr_actor, path_parent, _comp_path_func=comp_path_func
    )


def trace_back(actor_id_1, actor_id_2, path_parent, _comp_path_func=None):
    """
    Builds a path of actor ids or movies,
    depending on the input and the value
    of the complement path function

    *transformed_data must be a dictionary
    mapping each actor to a set of tuples
    documenting each actor they worked with,
    and the movie in which they acted

    path_parent is a dictionary associating each node
    with its parent in a shortest path from actor_id_1
    to actor_id_2
    """
    # assert movie_path == bool(transformed_data), 'error, either transformed_data \
    #     is provided when not necessary, or transformed data is not \
    #     provided when movie_path parameter requires it'

    curr_actor = actor_id_2
    path_of_actors = [curr_actor]

    if _comp_path_func is not None:
        complement_path = []

    while curr_actor != actor_id_1:
        curr_actor = path_parent[curr_actor]
        path_of_actors.append(curr_actor)

        if _comp_path_func is not None:
            complement_path.append(
                _comp_path_func(
                    (path_of_actors[-2], path_of_actors[-1])
                )
            )

    if _comp_path_func is not None:
        return complement_path[::-1]

    return path_of_actors[::-1]


def actors_connecting_films(transformed_data, film1, film2):
    """
    Finds the shortest actor path connecting two movies

    Accomplishes this by calling actor_path with movie
    ids passed in, and the right get_neighbors function

    *transformed_data must be a dictionary
    mapping each actor to a set of tuples
    documenting each actor they worked with,
    and the movie in which they acted

    Returns list of actor ids
    """
    # 1) use transformed_data to find relationship
    # between movies and associated actors
    # 2) use movie-to-actor data with
    # actor-to-actor+movie data to find movie neighbors
    # 3) use the breadth-first search to find
    # shortest path between movies
    # 4) find the associated chain of actors

    # dict: movie -> all actors
    movie_actor_data = get_movie_actor_data(transformed_data)

    def movie_neighbors_func(curr_movie, t_data, checked):
        # use transformed_data to find relationship between
        # movies and associated actors then use actor-to-actor
        # data to find relationship between actors and movies
        # if not in checked, each of these movies is a neighbor
        actor_neighbors = movie_actor_data[curr_movie]
        movie_neighbors = {
            neighbor[1]
            for actor in actor_neighbors
            for neighbor in t_data[actor]
            if neighbor[1] not in checked
        }
        return movie_neighbors

    def actor_path_element(last_two_nodes):
        # finds an actor that is shared
        # by both movies in the shortest path
        actor_set1 = movie_actor_data[last_two_nodes[1]]
        actor_set2 = movie_actor_data[last_two_nodes[0]]
        return list(actor_set1.intersection(actor_set2))[0]

    # use actor_path, but pass-in movie ids, have
    # a different neighbor finding algorithm, and use
    # a specialized complementary path function that
    # ensures that a list of actors that make-up the
    # path is what is returned
    return actor_path(
        transformed_data,
        film1,
        lambda x: (x == film2),
        get_neighbors_func=movie_neighbors_func,
        comp_path_func=actor_path_element,
    )


def get_movie_actor_data(transformed_data):
    """
    Takes in transformed data, which maps
    actors to other actors they worked with,
    and the associated movie, and returns an
    equivalent data structure which maps
    movies to associated actors
    """
    movie_actor_data = {}
    for actor in transformed_data.keys():
        for neighbor in transformed_data[actor]:
            movie_actor_data.setdefault(neighbor[1], set()).update([actor, neighbor[0]])
    return movie_actor_data


if __name__ == "__main__":
    with open("resources/names.pickle", "rb") as f:
        _names_db = pickle.load(f)  # loads a dictionary with actor-id key-value pairs

    # type of names_db and of its keys and values
    print(type(_names_db))
    print(type(_names_db.keys()))
    print(type(_names_db.values()))

    # Lisa Delien's name
    _id = _names_db["Lisa Delien"]
    print(f"Lisa Delien's id: {_id}")

    # Name for id 1169414
    for _name in _names_db.keys():
        if _names_db[_name] == 1169414:
            print(f"The id 1169414 is {_name}'s")
            break

    # --------------------------------------------------

    with open("resources/large.pickle", "rb") as f:
        large_db = pickle.load(f)  # loads a list of actor-actor-movie tuple triples

    actor_1 = name_to_id(_names_db, "Corinne Parquet")
    actor_2 = name_to_id(_names_db, "Bruce McGill")

    actor_ids = actor_to_actor_path(transform_data(large_db), actor_1, actor_2)
    actor_list = [id_to_name(_names_db, act_id) for act_id in actor_ids]
    print(actor_list)

    # --------------------------------------------------

    with open("resources/movies.pickle", "rb") as f:
        _movies_db = pickle.load(f)  # loads a dictionary of movie names to ids

    actor_1 = name_to_id(_names_db, "Kevin Bacon")
    actor_2 = name_to_id(_names_db, "Julia Roberts")

    movie_ids = movie_path(transform_data(large_db), actor_1, actor_2)
    movie_list = [id_to_movie(_movies_db, mov_id) for mov_id in movie_ids]
    print(movie_list)

    with open("resources/tiny.pickle", "rb") as f:
        tiny_db = pickle.load(f)  # loads a list of actor-actor-movie tuple triples

    print(tiny_db)

    # actor_ids = bacon_path(transform_data(large_db), name_to_id(names_db, 'Awie'))
    # actor_path = [id_to_name(names_db, _id) for _id in actor_ids]

    # actor1 = id_to_name(names_db, 'Beatrice Winde')
    # actor2 = id_to_name(names_db, 'David Clennon')

    # print(acted_together(transform_data(small_db), actor1, actor2))

    # additional code here will be run only when lab.py is invoked directly
    # (not when imported from test.py), so this is a good place to put code
    # used, for example, to generate the results for the online questions.
