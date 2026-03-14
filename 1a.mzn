%Definition of variables
var 1..4: x1;
var 1..3: x2;
%Definition of constraints
constraint x1>x2;
constraint x2=3;

%Solve problem
solve satisfy;