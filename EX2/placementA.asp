% a cell is the intersection of the R-th row and the C-th column
cell(R,C) :- row(R), col(C).

% a cell constains a wall, if containWall is derivable
% the capacity of the wall is not taken into account
containWall(R,C) :- wall(R,C,_).

% every cell around a wall can contain a lightbulb, if that cell is directly adjecent to the wall and the adjecent cell does not contain another wall
neighbouringWallCell(WR,WC,R,C) :- wall(WR,WC,_), cell(R,C), not containWall(R,C),
                               R - WR <= 1, WR - R <= 1,
                               C - WC <= 1, WC - C <= 1.

% a placabe cell, is one that neighbors a wall and can contain a lightbulb
placableCell(R,C) :- neighbouringWallCell(_,_,R,C).

% guess the placements of lightbulbs around walls
{ placeLightBulb(R,C) } :- placableCell(R,C).

% lightbulbs are not allowed to be placed in the same cell, where a wall is located
:- placeLightBulb(R,C), containWall(R,C).
% lightbulbs are not allowed to be placed on cells, that are not directly adjecent to a wall
:- placeLightBulb(R,C), not placableCell(R,C).

% a row is obstructed, if a wall is located between the start and end of the row
obstructedRow(R,C1,C2) :- row(R), col(C1), col(C2), wall(R,WC,_), C1 < WC, WC < C2.
obstructedRow(R,C1,C2) :- row(R), col(C1), col(C2), wall(R,WC,_), C2 < WC, WC < C1.

% a column is obstructed, if a wall is located between the start and end of the column
obstructedCol(C,R1,R2) :- col(C), row(R1), row(R2), wall(WR,C,_), R1 < WR, WR < R2.
obstructedCol(C,R1,R2) :- col(C), row(R1), row(R2), wall(WR,C,_), R2 < WR, WR < R1.

% a lightbulb must light up it's own cell and all the cells in a vertical and horizontal path, if those paths are not obstructed by a lightbulb
lighted(R,C) :- placeLightBulb(R,C).
lighted(R,C) :- cell(R,C), placeLightBulb(R,CB), C != CB, not obstructedRow(R,C,CB).
lighted(R,C) :- cell(R,C), placeLightBulb(RB,C), R != RB, not obstructedCol(C,R,RB).

% every cell, that does not contain a wall must be lit up
:- cell(R,C), not containWall(R,C), not lighted(R,C).

#show placeLightBulb/2.
