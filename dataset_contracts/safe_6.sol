// voting_contract.sol
pragma solidity ^0.4.15;

contract VotingContract {
    mapping (address => bool) public hasVoted;
    mapping (uint => uint) public votes;

    function vote(uint candidate) public {
        require(!hasVoted[msg.sender]);
        votes[candidate] += 1;
        hasVoted[msg.sender] = true;
    }

    function getVotes(uint candidate) public constant returns (uint) {
        return votes[candidate];
    }
}