#include <iostream>

#include <string>

//No longer need to use std::cout
using namespace std;

// Preprocessor to define Constants
#define PI 3.14159
#define NEWLINE '\n'

int main() {
     cout << "Hello C++ from CLion!" << '\n';
     cout << "This " << " is a " << "single C++ statement" << NEWLINE;

     cout << "Variables" << NEWLINE;;
     int income=5000;          // initial value: 5k
     int expense(3000);        // initial value: 3k
     int investment{1000};     // initial value: 2k
     int afterExpenses;        // initial value undetermined

     float loanRate (6.02e23f);    // float
     double interest (3.14159L );     // long double

     afterExpenses = expense + investment;
     afterExpenses = income - expense;
     cout << "After Expenses: " << afterExpenses;

     // Auto declear type
     auto saving = afterExpenses;  // the same as: int leisure = saving;

    // decltype based on value
    decltype(saving) freeMoney;  // same as : int freeMoney;

    // Character
     const char complexNumber = 'z';
     const char plankConstant = 'h';

    // String


    string userName = "John Doe";
    string userAddress ("Central Plaza, Munich");
    string useEmail {"john.doe@gmail.com"};

     cout << NEWLINE;
     cout << "I am " << userName << " my address is " << userAddress;

     // Constants
     const double pi = 3.14159;
     const char newline = '\n';

     double r=1.0;
     double circleArea;
     circleArea = PI * r * r;
     cout << circleArea;
     cout << NEWLINE;


     // Taking Input
      int age;
      cout << "Please enter your age: ";
      cin >>age;
      cout << "The age you entered is " <<age;


     return 0;
}