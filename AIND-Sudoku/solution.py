assignments = []
rows = 'ABCDEFGHI'
cols = '123456789'

def cross(A, B):
#    print('cross')
    "Cross product of elements in A and elements in B."
    return [a+b for a in A for b in B]

boxes = cross(rows, cols)

row_units = [cross(r, cols) for r in rows]
column_units = [cross(rows, c) for c in cols]
square_units = [cross(rs, cs) for rs in ('ABC','DEF','GHI') for cs in ('123','456','789')]
frontLeft_diagonal_unit = [[rows[i]+cols[i] for i in range(len(rows))]]
backRight_diagonal_unit = [[rows[i]+cols[::-1][i] for i in range(len(rows))]]
unitlist = row_units + column_units + square_units + frontLeft_diagonal_unit + backRight_diagonal_unit
units = dict((s, [u for u in unitlist if s in u]) for s in boxes)
peers = dict((s, set(sum(units[s],[]))-set([s])) for s in boxes)

def assign_value(values, box, value):
    """
    Please use this function to update your values dictionary!
    Assigns a value to a given box. If it updates the board record it.
    """
    #print('assign_value')
    # Don't waste memory appending actions that don't actually change any values
    if values[box] == value:
        return values

    values[box] = value
    if len(value) == 1:
        assignments.append(values.copy())
    return values

def naked_twins(values):
    """Eliminate values using the naked twins strategy.
    Args:
        values(dict): a dictionary of the form {'box_name': '123456789', ...}

    Returns:
        the values dictionary with the naked twins eliminated from peers.
    """
    #print('naked_twins')
    # check each unit    
    for unit in unitlist:
        twin_values=[values[s] for s in unit ]
        values_count=[twin_values.count(s) for s in twin_values]
        #find the naked twins boxes
        naked_twins = []
        for i in range(len(twin_values)):
            if len(twin_values[i]) == 2 and values_count[i] == 2:
                    naked_twins.append(unit[i])
                
        #find the non twin boxes
        non_twins=set(unit)-set(naked_twins)
        #iterate all the naked_twins in case there are multiple pairs
        for box in naked_twins:
            for digit in values[box]:
                for n in non_twins:
                    if len(values[n])>1:
                        assign_value(values,n,values[n].replace(digit,''))
                        #values[n]=values[n].replace(digit,'')
        
    # potential_twins = [box for box in values.keys() if len(values[box]) == 2]
    # # Collect boxes that have the same elements
    # naked_twins = [[box1,box2] for box1 in potential_twins \
    #                 for box2 in peers[box1] \
    #                 if set(values[box1])==set(values[box2]) ]
    return values
    
    

def grid_values(grid):
    """
    Convert grid into a dict of {square: char} with '123456789' for empties.
    Args:
        grid(string) - A grid in string form.
    Returns:
        A grid in dictionary form
            Keys: The boxes, e.g., 'A1'
            Values: The value in each box, e.g., '8'. If the box has no value, then the value will be '123456789'.
    """
    #print('grid_values')
    assert len(grid) == 81
    all_digits = '123456789'
    values = []
    for c in grid:
        if c == '.':
            values.append(all_digits)
        elif c in all_digits:
            values.append(c)
    return dict(zip(boxes,values))

def display(values):
    """
    Display the values as a 2-D grid.
    Args:
        values(dict): The sudoku in dictionary form
    """
    #print('display')
    width = 1 + max( len(values[s]) for s in boxes)
    line = '+'.join( ['-' * (width * 3) ] * 3)
    for r in rows:
        print(''.join(values[r+c].center(width)+('|'  if  c in '36' else '' )for c in cols))
        if r in 'CF': print(line)
    return

def eliminate(values):
    #print('eliminate')
    solved_values = [box for box in values.keys() if len(values[box]) == 1]
    for solved_value in solved_values:
        digit = values[solved_value]
        for peer in peers[solved_value]:
            values = assign_value(values,peer,values[peer].replace(digit,''))
    return values

def only_choice(values):
    #print('only_choice')
    choice_list = '123456789'
    for unit in unitlist:
        for digit in choice_list:
            digit_places =  [box for box in unit if digit in values[box]]
            if len(digit_places) == 1:
                values = assign_value(values, digit_places[0], digit)
    return values

def reduce_puzzle(values):
    #solved_values =  [box for box in values.keys() if len(values[box]) == 1]
    stalled = False
    while not stalled:
        # print('To Solve:')
        # display(values)
        solved_values_before = len([box for box in values.keys() if len(values[box]) == 1 ])
        # print('Eliminate')
        values = eliminate(values)
        # display(values)
        # print('Only Choice')
        values = only_choice(values)
        # display(values)
        # print('Naked Twins')
        values = naked_twins(values)
        # display(values)
        # print('')
        solved_values_after = len([box for box in values.keys() if len(values[box]) == 1 ])
        stalled = solved_values_after == solved_values_before 
        
        if len([box for box in values.keys() if len(values [box]) == 0]):
            return False
    return values

def search(values):
    values = reduce_puzzle(values)
    if values is False:
         return False
    if all(len(values[s]) == 1 for s in boxes):
        return values
    list_possibility = [k for k in values.keys() if len(values.get(k))== min([len(n) for n in values.values() if len(n)>1])]
    key = list_possibility[0]
    # Now use recurrence to solve each one of the resulting sudokus, and 
    for value in values[key]:
        trial_values = values.copy()
        trial_values[key] = value
        is_success  = search(trial_values)
        if is_success :
            return is_success 
def solve(grid):
    """
    Find the solution to a Sudoku grid.
    Args:
        grid(string): a string representing a sudoku grid.
            Example: '2.............62....1....7...6..8...3...9...7...6..4...4....8....52.............3'
    Returns:
        The dictionary representation of the final sudoku grid. False if no solution exists.
    """
    values = grid_values(grid)
    return search(values)

if __name__ == '__main__':
    #diag_sudoku_grid = '2.............62....1....7...6..8...3...9...7...6..4...4....8....52.............3'
    diag_sudoku_grid = '1......2.....9.5...............8...4.........9..7123...........3....4.....936.4..'
    naked_twins_sudoku_grid = '.......41......89...7....3........8.....47..2.......6.7.2........1.....4..6.9.3..'
    display(solve(diag_sudoku_grid))
    #display(solve(naked_twins_sudoku_grid))

    try:
        from visualize import visualize_assignments
        visualize_assignments(assignments)

    except SystemExit:
        pass
    except:
        print('We could not visualize your board due to a pygame issue. Not a problem! It is not a requirement.')
