pragma solidity ^0.4.15;

contract Vulnerable {
    mapping (address => uint) public userBalance;

    event LogDeposit(address user, uint amount);
    event LogWithdraw(address user, uint amount, bool success);

    function deposit() public payable {
        userBalance[msg.sender] += msg.value;
        LogDeposit(msg.sender, msg.value);
    }

    function withdraw() public {
        uint amount = userBalance[msg.sender];
        if (msg.sender.call.value(amount)()) {
            userBalance[msg.sender] = 0;
            LogWithdraw(msg.sender, amount, true);
        } else {
            LogWithdraw(msg.sender, amount, false);
        }
    }
}