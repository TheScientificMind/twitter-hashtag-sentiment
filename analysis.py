hashtag = input("What hashtag would you like to analyze (e.g. #photography): ").lower().strip()

# keeps asking until given a proper hashtag
while not hashtag[1:].isalnum() or hashtag[1:].isnumeric() or "#" not in hashtag:
    print("Your hashtag was invalid. Please input a new one.")
    hashtag = input("What new hashtag would you like to analyze (e.g. #photography): ")