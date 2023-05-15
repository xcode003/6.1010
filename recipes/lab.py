"""
6.1010 Spring '23 Lab 4: Recipes
"""

import pickle
import sys

sys.setrecursionlimit(20_000)
# NO ADDITIONAL IMPORTS!


def make_recipe_book(recipes):
    """
    Given recipes, a list containing compound and atomic food items, make and
    return a dictionary that maps each compound food item name to a list
    of all the ingredient lists associated with that name.
    """
    recipe_book = {}
    for item in recipes:
        if item[0] == "compound":
            recipe_book.setdefault(item[1], []).append(item[2])
    return recipe_book


def make_atomic_costs(recipes):
    """
    Given a recipes list, make and return a dictionary mapping each atomic food item
    name to its cost.
    """
    ingredients = {}
    for item in recipes:
        if item[0] == "atomic":
            ingredients[item[1]] = item[2]
    return ingredients


def lowest_cost(recipes, food_item, forbidden=None):
    """
    Given a recipes list and the name of a food item, return the lowest cost of
    a full recipe for the given food item.
    """
    recipe_book = make_recipe_book(recipes)
    atomic_costs = make_atomic_costs(recipes)

    # check output with new function use - good
    cheapest = recursive_cheapest_flat_recipe(
        atomic_costs, recipe_book, food_item, forbidden
    )

    if cheapest:  # not None
        return cheapest["price"]
    return cheapest


def scale_recipe(flat_recipe, n):
    """
    Given a dictionary of atomic ingredients mapped to quantities
    needed, returns a new dictionary with the quantities scaled by n.
    """
    return {ingredient: flat_recipe[ingredient] * n for ingredient in flat_recipe}


def make_grocery_list(flat_recipes):
    """
    Given a list of flat_recipe dictionaries that map atomic food items to quantities,
    return a new overall 'grocery list' dictionary that maps each atomic ingredient name
    to the sum of its quantities across the given flat recipes.

    For example,
        make_grocery_list([{'milk':1, 'chocolate':1}, {'sugar':1, 'milk':2}])
    should return:
        {'milk':3, 'chocolate': 1, 'sugar': 1}
    """
    grocery_list = {}
    for recipe in flat_recipes:
        for ingredient in recipe:
            if ingredient in grocery_list:
                grocery_list[ingredient] += recipe[ingredient]
            else:
                grocery_list[ingredient] = recipe[ingredient]
    return grocery_list


def cheapest_flat_recipe(recipes, food_item, forbidden=None):
    """
    Given a recipes list and the name of a food item, return a flat dictionary
    (mapping atomic food items to quantities) representing the cheapest full
    recipe for the given food item.

    Returns None if there is no possible recipe.
    """
    # given: a list of recipes; map of ingredient to (ingredient, quantity needed)

    # available functions
    # map of compound food -> list of recipes (ingredient, quantity)
    # map of atomic food -> price
    # scale flat recipe (ingredient -> quantity) by n
    # collapse list of flat recipes to a single flat recipe (ingredient -> quantity)

    # return: a map of atomic ingredients to quantities

    # code flow:
    # atomic and compound
    # atomic -> perfect, simply return map of ingredient to 1
    # compound -> perfect, just iterate through each recipe,
    # get atomic ingredients and scale, collapse appropriately
    # what if compound has compound ingredients? -> perfect,
    # scale and collapse the returned dictionary along with
    # the other ingredients, as if it were from an atomic call
    # (abstracted away in recursive calls)

    base_foods = make_atomic_costs(recipes)
    recipe_book = make_recipe_book(recipes)

    cheapest = recursive_cheapest_flat_recipe(
        base_foods, recipe_book, food_item, forbidden
    )
    if cheapest:  # not None
        return cheapest["recipe"]
    return cheapest


def recursive_cheapest_flat_recipe(
    atomic_foods, compound_recipes, food, forbidden_foods=None
):
    """
    Core recursive algorithm that finds the
    cheapest recipe; can be used to find the
    actual recipe, or its cost

    Returns a dictionary containing a
    the cheapest recipe and its cost
    """

    def recursive_helper(ingredient):
        """
        Helper function to mitigate the need
        to pass global variables in recursive calls
        """
        if forbidden_foods is None or ingredient not in forbidden_foods:
            if ingredient in atomic_foods:
                return {"recipe": {ingredient: 1}, "price": atomic_foods[ingredient]}
            elif ingredient in compound_recipes:
                recipe_options = []
                for recipe in compound_recipes[ingredient]:
                    recipe_price = 0
                    complete_recipe = []
                    impossible_recipe = False
                    for item in recipe:
                        item_info = recursive_helper(item[0])
                        if item_info is None:
                            impossible_recipe = True
                            break
                        complete_recipe.append(
                            scale_recipe(item_info["recipe"], item[1])
                        )
                        recipe_price += item_info["price"] * item[1]
                    if impossible_recipe is False:  # all ingredients have been found
                        complete_recipe = make_grocery_list(complete_recipe)
                        recipe_options.append(
                            {"recipe": complete_recipe, "price": recipe_price}
                        )
                if recipe_options:  # not empty
                    return min(recipe_options, key=lambda x: x["price"])
        # ingredient is forbidden, not present,
        # or it depends on another ingredient
        # that is not present
        return None

    return recursive_helper(food)


def ingredient_mixes(flat_recipes):
    """
    Given a list of lists of dictionaries, where
    each inner list represents all the flat recipes
    to make a certain ingredient as part of a larger
    recipe, compute all combinations of the flat recipes.
    """
    if len(flat_recipes) == 1:
        return flat_recipes[0]
    mixes = []
    # this must be a list of dictionaries -- good
    other_combinations = ingredient_mixes(flat_recipes[1:])
    for recipe_option in flat_recipes[0]:
        for other_recipe in other_combinations:
            # create new list with two dictionaries as elements
            temp = [recipe_option, other_recipe]
            mixes.append(make_grocery_list(temp))
    return mixes

    # [[recipe1, recipe2, recipe3], [second set], [etc.]]
    # always input list of list of dictionaries
    # always return list of recipe dictionaries


def flatten(recipes):
    """
    Takes in a list of lists of all recipe
    combinations, where each sub-list represents
    a single recipe for a parent ingredient

    Guaranteed to have only three nested
    collection data structures

    Returns a list of recipes (list of dictionaries)
    Returns [] if recipes is []
    """
    return [combo for rec in recipes for combo in rec]


def all_flat_recipes(recipes, food_item, forbidden_foods=None):
    """
    Given a list of recipes and the name of a food item, produce a list (in any
    order) of all possible flat recipes for that category.

    Returns an empty list if there are no possible recipes
    """
    atomic_foods = make_atomic_costs(recipes).keys()
    compound_recipes = make_recipe_book(recipes)

    def recursive_list_all_sub_recipes(ingredient):  # returns a list of dictionaries
        """
        Recursive helper function to create a list containing
        each flat dictionary recipe for an ingredient
        (in the overall recipe to make food_item)
        """
        # first, get it to work with no excluded foods
        if forbidden_foods is None or ingredient not in forbidden_foods:
            # only one list
            if ingredient in atomic_foods:
                return [{ingredient: 1}]
            # construct list of recipes for ingredient
            elif ingredient in compound_recipes:
                every_recipe = []
                for recipe in compound_recipes[ingredient]:
                    all_item_recipes = []
                    impossible_recipe = False
                    for item in recipe:
                        sub_recipes = recursive_list_all_sub_recipes(item[0])
                        scaled_sub_recipes = [
                            scale_recipe(rec, item[1]) for rec in sub_recipes
                        ]
                        if scaled_sub_recipes == []:
                            impossible_recipe = True
                            break
                        all_item_recipes.append(scaled_sub_recipes)
                    if not impossible_recipe:
                        # if food_item depends on a
                        # forbidden food in a  given
                        # recipe, that recipe will
                        # not be added to every_recipe
                        every_recipe.append(ingredient_mixes(all_item_recipes))
                # if food_item cannot be made
                # because all recipes contain
                # forbidden foods, flatten here
                # will return an empty list
                return flatten(every_recipe)
        # if food item is forbidden
        return []

    return recursive_list_all_sub_recipes(food_item)


if __name__ == "__main__":
    # load example recipes from section 3 of the write-up
    with open("test_recipes/example_recipes.pickle", "rb") as f:
        example_recipes = pickle.load(f)

    cookie_recipes = [
        ("compound", "cookie sandwich", [("cookie", 2), ("ice cream scoop", 3)]),
        ("compound", "cookie", [("chocolate chips", 3)]),
        ("compound", "cookie", [("sugar", 10)]),
        ("atomic", "chocolate chips", 200),
        ("atomic", "sugar", 5),
        ("compound", "ice cream scoop", [("vanilla ice cream", 1)]),
        ("compound", "ice cream scoop", [("chocolate ice cream", 1)]),
        ("atomic", "vanilla ice cream", 20),
        ("atomic", "chocolate ice cream", 30),
    ]

    # print(all_flat_recipes(cookie_recipes, "cookie sandwich"))

    # [{'chocolate chips':6, 'vanilla ice cream':3},
    #  {'sugar':20, 'vanilla ice cream':3},
    #  {'chocolate chips':6, 'chocolate ice cream':3},
    #  {'sugar':20, 'chocolate ice cream':3}]

    recipes = [
    ('compound', 'chili',
    [
    ('beans', 3),
    ('cheese', 10),
    ('chili powder', 1),
    ('cornbread', 2),
    ('protein', 1)
    ]),
    ('atomic', 'beans', 5),
    ('compound', 'cornbread',
    [
    ('cornmeal', 3),
    ('milk', 1),
    ('butter', 5),
    ('salt', 1),
    ('flour', 2)
    ]),
    ('atomic', 'cornmeal', 7.5),
    ('compound', 'burger',
    [
    ('bread', 2),
    ('cheese', 1),
    ('lettuce', 1),
    ('protein', 1),
    ('ketchup', 1)
    ]),
    ('compound', 'burger',
    [
    ('bread', 2),
    ('cheese', 2),
    ('lettuce', 1),
    ('protein', 2)
    ]),
    ('atomic', 'lettuce', 2),
    ('compound', 'butter',
    [
    ('milk', 1),
    ('butter churn', 1)
    ]),
    ('atomic', 'butter churn', 50),
    ('compound', 'milk', [('cow', 1), ('milking stool', 1)]),
    ('compound', 'cheese', [('milk', 1), ('time', 1)]),
    ('compound', 'cheese', [('cutting-edge laboratory', 11)]),
    ('atomic', 'salt', 1),
    ('compound', 'bread', [('yeast', 1), ('salt', 1), ('flour', 2)]),
    ('compound', 'protein', [('cow', 1)]),
    ('atomic', 'flour', 3),
    ('compound', 'ketchup', [('tomato', 30), ('vinegar', 5)]),
    ('atomic', 'chili powder', 1),
    ('compound', 'ketchup',
    [
    ('tomato', 30),
    ('vinegar', 3),
    ('salt', 1),
    ('sugar', 2),
    ('cinnamon', 1)
    ]),  # the fancy ketchup
    ('atomic', 'cow', 100),
    ('atomic', 'milking stool', 5),
    ('atomic', 'cutting-edge laboratory', 1000),
    ('atomic', 'yeast', 2),
    ('atomic', 'time', 10000),
    ('atomic', 'vinegar', 20),
    ('atomic', 'sugar', 1),
    ('atomic', 'cinnamon', 7),
    ('atomic', 'tomato', 13),
    ]

    print(all_flat_recipes(recipes, 'chili'))
