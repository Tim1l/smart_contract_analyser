// token_contract.sol
pragma solidity ^0.4.15;

contract TokenContract {
    mapping (address => uint) public balances;

    function mint(address recipient, uint amount) public {
        balances[recipient] += amount;
    }

    function transfer(address recipient, uint amount) public {
        require(balances[msg.sender] >= amount);
        balances[msg.sender] -= amount;
        balances[recipient] += amount;
    }

    function getBalance(address account) public constant returns (uint) {
        return balances[account];
    }
}