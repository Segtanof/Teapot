import random
import time
from typing import Tuple, Optional

class RockPaperScissors:
    def __init__(self):
        self.choices = ['rock', 'paper', 'scissors']
        self.valid_inputs = {
            'r': 'rock',
            'rock': 'rock',
            'p': 'paper',
            'paper': 'paper',
            's': 'scissors',
            'scissors': 'scissors'
        }
        self.player_score = 0
        self.computer_score = 0
        self.ties = 0

    def get_player_choice(self) -> Optional[str]:
        """Get and validate player's choice"""
        while True:
            choice = input("\nEnter your choice (r/p/s or rock/paper/scissors) or 'q' to quit: ").lower().strip()
            
            if choice == 'q':
                return None
            
            if choice in self.valid_inputs:
                return self.valid_inputs[choice]
            
            print("Invalid choice! Please try again.")

    def get_computer_choice(self) -> str:
        """Generate computer's choice with a dramatic pause"""
        print("\nComputer is choosing", end="")
        for _ in range(3):
            print(".", end="", flush=True)
            time.sleep(0.3)
        print()
        
        return random.choice(self.choices)

    def determine_winner(self, player_choice: str, computer_choice: str) -> Tuple[str, str]:
        """Determine the winner and return result message"""
        if player_choice == computer_choice:
            self.ties += 1
            return "It's a tie!", "🤝"
        
        winning_combinations = {
            'rock': 'scissors',
            'paper': 'rock',
            'scissors': 'paper'
        }
        
        if winning_combinations[player_choice] == computer_choice:
            self.player_score += 1
            return "You win!", "🎉"
        else:
            self.computer_score += 1
            return "Computer wins!", "💻"

    def display_choices(self, player_choice: str, computer_choice: str) -> None:
        """Display both choices with emoji"""
        emoji_map = {
            'rock': '🪨',
            'paper': '📄',
            'scissors': '✂️'
        }
        print(f"\nYou chose: {player_choice} {emoji_map[player_choice]}")
        print(f"Computer chose: {computer_choice} {emoji_map[computer_choice]}")

    def display_score(self) -> None:
        """Display current score"""
        print("\n=== Current Score ===")
        print(f"You: {self.player_score} 👤")
        print(f"Computer: {self.computer_score} 💻")
        print(f"Ties: {self.ties} 🤝")
        print("==================")

    def play_game(self) -> None:
        """Main game loop"""
        print("\n=== Welcome to Rock Paper Scissors! ===")
        print("Can you beat the computer? Let's find out!")
        
        while True:
            player_choice = self.get_player_choice()
            
            if player_choice is None:
                break
                
            computer_choice = self.get_computer_choice()
            self.display_choices(player_choice, computer_choice)
            
            result, emoji = self.determine_winner(player_choice, computer_choice)
            print(f"\n{result} {emoji}")
            
            self.display_score()

        # Display final results
        print("\n=== Final Results ===")
        print(f"Games played: {self.player_score + self.computer_score + self.ties}")
        print(f"Your wins: {self.player_score}")
        print(f"Computer wins: {self.computer_score}")
        print(f"Ties: {self.ties}")
        
        if self.player_score > self.computer_score:
            print("\nCongratulations! You're the overall winner! 🏆")
        elif self.computer_score > self.player_score:
            print("\nComputer is the overall winner! Better luck next time! 🤖")
        else:
            print("\nIt's a tie overall! Great game! 🤝")
        
        print("\nThanks for playing! 👋")

if __name__ == "__main__":
    game = RockPaperScissors()
    game.play_game()