% --- Base Word Definitions (Facts) ---

% Units (1-9)
unit(one, 1).
unit(two, 2).
unit(three, 3).
unit(four, 4).
unit(five, 5).
unit(six, 6).
unit(seven, 7).
unit(eight, 8).
unit(nine, 9).

% Teens (10-19)
teen(ten, 10).
teen(eleven, 11).
teen(twelve, 12).
teen(thirteen, 13).
teen(fourteen, 14).
teen(fifteen, 15).
teen(sixteen, 16).
teen(seventeen, 17).
teen(eighteen, 18).
teen(nineteen, 19).

% Tens (20, 30, ..., 90)
tens(twenty, 20).
tens(thirty, 30).
tens(forty, 40).
tens(fifty, 50).
tens(sixty, 60).
tens(seventy, 70).
tens(eighty, 80).
tens(ninety, 90).


% Converts an English word string (e.g., "ninety nine") into a number.
% WordString: Input string of English number words.
% Number: Output numerical value.
to_num(WordString, Number) :-
    string_lower(WordString, LowerWordString),            % Convert to lowercase
    split_string(LowerWordString, " ", "", StringWordsList), % Split by space, ignore empty parts
    maplist(atom_string, AtomWordsList, StringWordsList), % Convert list of strings to list of atoms

    % 2. Use DCG to parse the list of atoms and calculate the number
    phrase(number_phrase(Number), AtomWordsList, []),
    Number =< 1000. % Ensure the resulting number is within the constraint

% --- DCG Rules for Parsing Number Words ---

% Case 1: "one thousand"
number_phrase(1000) --> [one, thousand].

% Case 2: Numbers with a "hundred" component (e.g., "two hundred and fifty three", "five hundred")
number_phrase(Value) -->
    unit_word(H),             % e.g., "two"
    [hundred],                % "hundred"
    (   % After "hundred", there can be "and <tens_units>", or just "<tens_units>", or nothing.
        ( [and], tens_units_val(TU) )     % "and fifty three"
    ;   ( tens_units_val(TU) )            % "fifty three" (without "and" - less common but possible)
    ;   ( {TU = 0} )                      % Nothing follows "hundred" (e.g. "two hundred")
    ),
    { Value is H * 100 + TU }.


% Case 3: Numbers without a "hundred" component (0-99) (e.g., "forty two", "seven")
number_phrase(Value) -->
    tens_units_val(Value).


% e.g., "forty five", "twenty one"
tens_units_val(Value) -->
    tens_word(T),
    unit_word(U),
    { Value is T + U }.

% e.g., "twenty", "forty" (as a whole number, or part of "four hundred and twenty")
tens_units_val(Value) -->
    tens_word(Value).

% e.g., "thirteen", "nineteen"
tens_units_val(Value) -->
    teen_word(Value).

% e.g., "five", "nine"
tens_units_val(Value) -->
    unit_word(Value).


% --- Word Category DCG Rules ---
% These rules consume one word from the list and check if it belongs to a category.

unit_word(Value) --> [Word], { unit(Word, Value) }.
teen_word(Value) --> [Word], { teen(Word, Value) }.
tens_word(Value) --> [Word], { tens(Word, Value) }.




/*
% --- Testing Examples ---
?- to_num("ninety nine", N).
N = 99.

?- to_num("one hundred and one", N).
N = 101.

?- to_num("three hundred and ninety four", N).
N = 394.

?- to_num("forty five", N).
N = 45.

?- to_num("one thousand", N).
N = 1000.

?- to_num("six hundred", N).
N = 600.

?- to_num("seven", N).
N = 7.

?- to_num("seventeen", N).
N = 17.

?- to_num("eighty", N).
N = 80.

% Edge case: "one hundred ten" (without "and")
?- to_num("one hundred ten", N).
N = 110.

% Constraint check (N <= 1000)
% ?- to_num("one thousand and one", N). % Should fail due to constraint or grammar
% This specific grammar will fail because "one thousand" is terminal.
% If we wanted to parse "one thousand and one", the grammar for numbers > 1000 would be more complex.
% Current behavior:
% ?- phrase(number_phrase(N), [one, thousand, and, one], []).
% false. (Correct, as our grammar doesn't support this)

% ?- to_num("zero", N). % "zero" is not defined in units, will fail. Correct.
% false.

% ?- to_num("Twenty Two", N). % Test case-insensitivity
% N = 22.
*/