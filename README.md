# Solitaire-Curriculum-Generation

Based on the following (called High's Card Game):

You will be given 30 unique cards from a standard deck of 52 cards. With these cards you will form a 5x5 grid (meaning there are 5 cards you will not be using). 
Your goal is to have as many combos as possible in your rows, columns and your two diagonals. 

I will be looking for the following combos in your rows, columns and diagonals:
- Pairs, triples, quadruples. The cards do not necessarily need to be next to each other for it to count (ex. 57953 is still a pair of 5s)
- 5-card straight (8 9 10 J Q for instance). The cards do not need to be in order, so something like 59687 would be just as valid. 
- 5-card flush (all 5 cards are of the same suit) 

I will scan each row, column and diagonal. Cards that aren't part of a combo gives points. 
If a card is part of a combo in its row, but not part of anything in its column, it will still give you points (for the column). 
Points are bad, you do not want points.

Cards have the following point values, if they end up not fitting in a combo:
Aces are worth 15 points each
10 to King are worth 10 points each
2-9 are worth the number it says. So a 4 is worth 4 points. 
