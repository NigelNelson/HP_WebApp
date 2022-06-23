// import Skeleton from 'react-loading-skeleton'
// import 'react-loading-skeleton/dist/skeleton.css'

class Image extends React.Component {
    render() {

        this.state = {
            margin: "auto"
        }

        return (
            <div id={this.props.id} className={"img-thumbnail"}></div>
        )
    }
}