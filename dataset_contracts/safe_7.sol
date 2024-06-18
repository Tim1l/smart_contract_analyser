// loyalty_program.sol
pragma solidity ^0.4.15;

contract LoyaltyProgram {
    struct Member {
        uint points;
        uint level;
    }

    mapping (address => Member) public members;

    function earnPoints(address member, uint points) public {
        members[member].points += points;
        updateLevel(member);
    }

    function redeemPoints(address member, uint points) public {
        require(members[member].points >= points);
        members[member].points -= points;
    }

    function updateLevel(address member) internal {
        if (members[member].points > 1000) {
            members[member].level = 3;
        } else if (members[member].points > 500) {
            members[member].level = 2;
        } else {
            members[member].level = 1;
        }
    }

    function getPoints(address member) public constant returns (uint) {
        return members[member].points;
    }

    function getLevel(address member) public constant returns (uint) {
        return members[member].level;
    }
}