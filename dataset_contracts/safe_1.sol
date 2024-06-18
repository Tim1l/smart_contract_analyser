// safe_contract_1.sol
pragma solidity ^0.4.15;

contract SafeContract1 {
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