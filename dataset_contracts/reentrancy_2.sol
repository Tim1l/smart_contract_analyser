// vulnerable_reentrancy.sol
pragma solidity ^0.4.15;

contract VulnerableReentrancy {
    mapping (address => uint) public userBalance;

    function deposit() public payable {
        userBalance[msg.sender] += msg.value;
    }

    function withdraw(uint amount) public {
        require(userBalance[msg.sender] >= amount);
        msg.sender.call.value(amount)();
        userBalance[msg.sender] -= amount;
    }

    function getBalance() public constant returns (uint) {
        return address(this).balance;
    }
}