// safe_contract_2.sol
pragma solidity ^0.4.15;

contract SafeContract2 {
    mapping (address => uint) public userBalance;

    function deposit() public payable {
        userBalance[msg.sender] += msg.value;
    }

    function withdraw(uint amount) public {
        require(userBalance[msg.sender] >= amount);
        userBalance[msg.sender] -= amount;
        msg.sender.transfer(amount);
    }

    function getBalance() public constant returns (uint) {
        return address(this).balance;
    }
}